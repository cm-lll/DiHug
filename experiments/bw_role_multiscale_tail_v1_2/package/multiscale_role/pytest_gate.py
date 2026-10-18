"""Exact frozen collection + all three passed phases; no skipped/xfail cases."""
import hashlib
import json
from pathlib import Path
from multiscale_role.runtime_guard import atomic_json


def contract(root):
    root=Path(root);spec=json.loads((root/'TEST_SUITE.json').read_text())
    ids=spec['nodeids']
    if not ids or len(ids)!=len(set(ids)) or len(ids)!=spec['total'] or spec['version']!='bw_role_multiscale_tail_v1_2':
        raise ValueError('invalid frozen test collection')
    actual={str(p.relative_to(root)) for p in (root/'tests').rglob('*.py') if '__pycache__' not in p.parts}
    if actual!=set(spec['test_file_sha256']):raise ValueError('frozen test file set differs')
    for name,h in spec['test_file_sha256'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=h:raise ValueError('frozen test source differs: '+name)
    return spec


def complete(spec,collected,deselected,reports):
    expected=spec['nodeids']
    if sorted(collected)!=expected or len(collected)!=len(expected) or deselected:return False
    if set(reports)!=set(expected):return False
    return all(r=={'setup':'passed','call':'passed','teardown':'passed'} for r in reports.values())


class FullSuiteGate:
    def __init__(self,root,output):
        self.spec=contract(root);self.output=Path(output);self.collected=[];self.deselected=[];self.reports={}
    def pytest_collection_finish(self,session):self.collected=[i.nodeid for i in session.items]
    def pytest_deselected(self,items):self.deselected.extend(i.nodeid for i in items)
    def pytest_runtest_logreport(self,report):
        row=self.reports.setdefault(report.nodeid,{})
        if report.when in row:row['duplicate_phase']='failed'
        row[report.when]='xfail' if hasattr(report,'wasxfail') else report.outcome
    def pytest_sessionfinish(self,session,exitstatus):
        passed=exitstatus==0 and complete(self.spec,self.collected,self.deselected,self.reports)
        atomic_json(self.output,dict(status='complete' if passed else 'incomplete',
            expected_nodeids=self.spec['nodeids'],collected_nodeids=self.collected,deselected_nodeids=self.deselected,
            reports=self.reports,full_suite_passed=passed))
        if not passed:session.exitstatus=1


def validate(root,out):
    import xml.etree.ElementTree as ET
    spec=contract(root);out=Path(out);record=json.loads((out/'test_execution.json').read_text())
    if record.get('expected_nodeids')!=spec['nodeids'] or record.get('full_suite_passed') is not True or record.get('status')!='complete':
        raise ValueError('full test collection receipt missing or incomplete')
    if not complete(spec,record['collected_nodeids'],record['deselected_nodeids'],record['reports']):
        raise ValueError('test identities/phases incomplete, skipped, deselected or duplicated')
    suites=list(ET.parse(out/'tests.xml').getroot().iter('testsuite'))
    cases=list(ET.parse(out/'tests.xml').getroot().iter('testcase'))
    if sum(int(s.get('tests',0)) for s in suites)!=spec['total'] or len(cases)!=spec['total'] or any(int(s.get(k,0)) for s in suites for k in ('errors','failures','skipped')):
        raise ValueError('JUnit does not cover the frozen complete test suite')
    if any(c.find(k) is not None for c in cases for k in ('skipped','failure','error')):
        raise ValueError('JUnit contains unsuccessful test cases')
