"""Admission and exact-suite engineering negatives; no real diagnostic inputs."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
import pytest
from multiscale_role.entry_guard import validate_output_path,verify_sealed_ledger,SEALED_LEDGER_SHA256,test_environment as clean_env
from multiscale_role.pytest_gate import validate,complete

ROOT=Path(__file__).resolve().parents[1]


def snapshot(path):
    return {str(p.relative_to(path)):('directory' if p.is_dir() else hashlib.sha256(p.read_bytes()).hexdigest())
            for p in path.rglob('*')}


@pytest.mark.parametrize('location',['results','symlink','states','weights','short_input'])
def test_rejected_output_adds_no_file_to_protected_inputs(tmp_path,location):
    old=tmp_path/'old';old.mkdir();states=tmp_path/'states';states.mkdir();weights=tmp_path/'weights';weights.mkdir()
    (old/'activation.json').write_text(json.dumps(dict(states=str(states),checkpoints=str(weights))))
    (old/'evidence_hashes.json').write_text('{"fixture":"no scientific input"}')
    (states/'kept.bin').write_bytes(b'unchanged');(weights/'kept.bin').write_bytes(b'unchanged')
    if location=='symlink':(tmp_path/'alias').symlink_to(old,target_is_directory=True)
    parent={'results':old,'symlink':tmp_path/'alias','states':states,'weights':weights,'short_input':old}[location]
    out=parent/'must_not_be_created'/'nested'
    if location=='short_input':
        (old/'manifest.json').write_text('{"records":[]}');(old/'evaluation.json').write_text('{}')
        args=['short','--manifest',str(old/'manifest.json'),'--manifest-sha256','0'*64,
              '--evaluation',str(old/'evaluation.json'),'--evaluation-sha256','0'*64]
    else:args=['export-legacy','--legacy-run',str(old),'--legacy-zip',str(tmp_path/'not_read.zip'),
               '--legacy-ledger-sha256',SEALED_LEDGER_SHA256]
    before={str(p):snapshot(p) for p in (old,states,weights)}
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
    result=subprocess.run([sys.executable,str(ROOT/'run_multiscale.py'),*args,'--output',str(out),'--seconds','30'],
                          capture_output=True,text=True,env=env,timeout=10)
    assert result.returncode!=0 and 'overlaps protected' in result.stderr
    assert not out.parent.exists() and before=={str(p):snapshot(p) for p in (old,states,weights)}


def test_path_sibling_allowed_but_source_or_ancestor_rejected(tmp_path):
    source=tmp_path/'source';old=tmp_path/'old';old.mkdir();source.mkdir()
    receipt=validate_output_path('export-legacy',tmp_path/'new',source,legacy_run=old,legacy_zip=tmp_path/'uploaded.zip')
    assert receipt['output']==str(tmp_path/'new') and not (tmp_path/'new').exists()
    for out in (source/'child',tmp_path):
        with pytest.raises(ValueError,match='overlaps'):validate_output_path('export-legacy',out,source,legacy_run=old)


@pytest.mark.parametrize('expected_kind',['sealed_but_changed_file','self_computed_current_hash'])
def test_unsealed_or_changed_ledger_rejected_without_writes(tmp_path,expected_kind):
    old=tmp_path/'old';old.mkdir();ledger=old/'evidence_hashes.json';ledger.write_text('{}')
    expected=SEALED_LEDGER_SHA256 if expected_kind=='sealed_but_changed_file' else hashlib.sha256(ledger.read_bytes()).hexdigest()
    before=snapshot(old);out=tmp_path/'new_output'
    result=subprocess.run([sys.executable,str(ROOT/'run_multiscale.py'),'export-legacy','--legacy-run',str(old),
        '--legacy-zip',str(tmp_path/'not_read.zip'),'--legacy-ledger-sha256',expected,'--output',str(out),'--seconds','30'],
        capture_output=True,text=True,timeout=10)
    assert result.returncode!=0 and 'ledger' in result.stderr and not out.exists() and snapshot(old)==before


def test_correct_sealed_value_checked_not_substituted(monkeypatch,tmp_path):
    import multiscale_role.entry_guard as guard
    calls=[]
    def hashed(path,check):calls.append(path);check();return SEALED_LEDGER_SHA256
    monkeypatch.setattr(guard,'checked_sha',hashed)
    assert verify_sealed_ledger(tmp_path,SEALED_LEDGER_SHA256)==SEALED_LEDGER_SHA256 and len(calls)==1
    with pytest.raises(ValueError):verify_sealed_ledger(tmp_path,'0'*64)
    assert len(calls)==1 # refuse replacement expected hash even before reading


def test_inherited_pytest_selection_and_plugin_environment_isolated():
    result=clean_env(dict(PYTEST_ADDOPTS='-k test_contract --deselect something',PYTEST_PLUGINS='bad',
        PYTEST_DISABLE_PLUGIN_AUTOLOAD='0',PYTEST_CURRENT_TEST='parent',OMP_NUM_THREADS='1',KEEP='ok'))
    assert result==dict(PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',KEEP='ok')


def fake_suite(root,out,total=115):
    (root/'tests').mkdir(parents=True);source=root/'tests/test_fixture.py';source.write_text('# engineering receipt fixture only\n')
    ids=['tests/test_fixture.py::test_%03d'%i for i in range(total)]
    spec=dict(version='bw_role_multiscale_tail_v1_2',total=total,nodeids=ids,
              test_file_sha256={'tests/test_fixture.py':hashlib.sha256(source.read_bytes()).hexdigest()})
    (root/'TEST_SUITE.json').write_text(json.dumps(spec))
    receipt=dict(status='complete',expected_nodeids=ids,collected_nodeids=ids.copy(),deselected_nodeids=[],full_suite_passed=True,
                 reports={i:dict(setup='passed',call='passed',teardown='passed') for i in ids})
    suite=ET.Element('testsuite',tests=str(total),errors='0',failures='0',skipped='0')
    for i in ids:ET.SubElement(suite,'testcase',name=i)
    out.mkdir();ET.ElementTree(suite).write(out/'tests.xml');(out/'test_execution.json').write_text(json.dumps(receipt))
    return spec,receipt,suite


@pytest.mark.parametrize('bad',['old_85','old_112','same_count_wrong_identity','duplicate','skip','xfail','missing_teardown','deselected','empty_receipt','junit_skip'])
def test_exact_collection_and_zero_skip_not_old_count_floor(tmp_path,bad):
    root=tmp_path/'package';out=tmp_path/'out';spec,record,xml=fake_suite(root,out)
    validate(root,out)
    ids=spec['nodeids']
    if bad in ('old_85','old_112'):
        n=85 if bad=='old_85' else 112;record['collected_nodeids']=ids[:n];record['reports']={k:v for k,v in record['reports'].items() if k in ids[:n]}
    elif bad=='same_count_wrong_identity':record['collected_nodeids'][-1]='tests/test_fixture.py::unapproved_test'
    elif bad=='duplicate':record['collected_nodeids'][-1]=ids[0]
    elif bad in ('skip','xfail'):record['reports'][ids[0]]['call']='skipped' if bad=='skip' else 'xfail'
    elif bad=='missing_teardown':record['reports'][ids[0]].pop('teardown')
    elif bad=='deselected':record['deselected_nodeids']=[ids[0]]
    elif bad=='empty_receipt':record={}
    else:ET.SubElement(xml[0],'skipped');ET.ElementTree(xml).write(out/'tests.xml')
    (out/'test_execution.json').write_text(json.dumps(record))
    with pytest.raises((ValueError,KeyError)):validate(root,out)


def test_manifest_checks_test_files_and_new_file_set(tmp_path):
    root=tmp_path/'package';out=tmp_path/'out';fake_suite(root,out)
    (root/'tests/test_added.py').write_text('# unrecorded file')
    with pytest.raises(ValueError,match='file set'):validate(root,out)
    (root/'tests/test_added.py').unlink();(root/'tests/test_fixture.py').write_text('# altered')
    with pytest.raises(ValueError,match='source differs'):validate(root,out)
