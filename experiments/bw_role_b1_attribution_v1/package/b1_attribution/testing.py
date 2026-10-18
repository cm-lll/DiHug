"""Reuse unchanged v1.2 full-suite phase/identity gate for B1 small tests."""
import hashlib
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET
from multiscale_role.pytest_gate import FullSuiteGate,complete


def spec(root):
    root=Path(root);s=json.loads((root/'B1_TEST_SUITE.json').read_text());ids=s['nodeids']
    if s['version']!='B1_attribution_v1' or not ids or ids!=sorted(set(ids)) or len(ids)!=s['total']:raise ValueError('invalid frozen B1 test suite')
    files={str(p.relative_to(root)) for p in (root/'b1_tests').rglob('*.py') if '__pycache__' not in p.parts}
    if files!=set(s['test_file_sha256']):raise ValueError('B1 test file set differs')
    for name,h in s['test_file_sha256'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=h:raise ValueError('B1 test source changed')
    return s


class Gate(FullSuiteGate):
    def __init__(self,root,out):
        self.spec=spec(root);self.output=Path(out)/'test_execution.json';self.collected=[];self.deselected=[];self.reports={}


def run_tests(root,out):
    os.chdir(root)
    import pytest
    return pytest.main(['-q','-c',str(Path(root)/'pytest.ini'),'-o','addopts=',
        '--confcutdir='+str(Path(root)/'b1_tests'),'b1_tests','--disable-warnings',
        '--junitxml='+str(Path(out)/'tests.xml')],plugins=[Gate(root,out)])


def validate_tests(root,out):
    s=spec(root);r=json.loads((Path(out)/'test_execution.json').read_text())
    if r.get('expected_nodeids')!=s['nodeids'] or r.get('full_suite_passed') is not True or not complete(s,r['collected_nodeids'],r['deselected_nodeids'],r['reports']):
        raise ValueError('B1 test collection/phases incomplete')
    xml=ET.parse(Path(out)/'tests.xml').getroot();suites=list(xml.iter('testsuite'))
    if len(list(xml.iter('testcase')))!=s['total'] or sum(int(v.get('tests',0)) for v in suites)!=s['total'] or any(int(v.get(k,0)) for v in suites for k in ('errors','failures','skipped')):
        raise ValueError('B1 JUnit incomplete')
