"""Public CLI/process tests and complete-release phase binding.

Only a built-in four-slot engineering fixture is used. No real/imitated six
state arrays, notebooks, models, weights, training, or scientific chain.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import pytest
from dynamic_role import runner

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('scenario,expected_code,expected_status,expected_reason',[
    ('success',0,'completed','completed'),
    ('invalid',1,'failed','child_failed'),
    ('timeout',124,'incomplete','timeout'),
    ('STOP',75,'stopped','STOP'),
])
def test_public_cli_parent_child_supervisor(scenario,expected_code,expected_status,expected_reason,tmp_path,record_property):
    out=tmp_path/'fresh_output';log=tmp_path/'public_cli.log'
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    mode='wait' if scenario in ('timeout','STOP') else scenario
    command=[sys.executable,str(ROOT/'run_dynamic.py'),'smoke','--smoke-mode',mode,
             '--output',str(out),'--max-seconds','20']
    start=time.monotonic()
    with log.open('w') as stream:
        proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        try:
            if scenario=='STOP':
                until=time.monotonic()+12
                while not (out/'smoke_started.json').is_file():
                    if proc.poll() is not None or time.monotonic()>until:
                        pytest.fail('real smoke child did not start: '+log.read_text())
                    time.sleep(.025)
                (out/'STOP').touch()
            code=proc.wait(timeout=35)
        finally:
            # Clean only the process this fixture owns if the test itself fails.
            if proc.poll() is None:
                if out.is_dir():(out/'STOP').touch()
                proc.terminate()
                try:proc.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    proc.kill();proc.wait(timeout=3)
                    if (out/'controller_child.json').exists():
                        import signal
                        pgid=json.loads((out/'controller_child.json').read_text())['pgid']
                        try:os.killpg(pgid,signal.SIGKILL)
                        except ProcessLookupError:pass
    assert code==expected_code,log.read_text()
    receipt=json.loads((out/'controller.json').read_text())
    assert receipt['exit_code']==code and receipt['status']==expected_status and receipt['reason']==expected_reason
    assert receipt['single_run_limit_seconds']==20 and all(receipt['cleanup'].values())
    child=json.loads((out/'controller_child.json').read_text())
    assert child['pid']!=proc.pid and child['pgid']==child['pid']
    with pytest.raises(ProcessLookupError):os.killpg(child['pgid'],0)
    artifact=out/'engineering_result.json'
    if scenario=='success':
        summary=json.loads((out/'summary.json').read_text());result=json.loads(artifact.read_text())
        assert summary['status']=='complete' and summary['scope']=='engineering_cli_smoke'
        assert summary['checks'] and all(summary['checks'].values())
        assert hashlib.sha256(artifact.read_bytes()).hexdigest()==summary['artifact_sha256']
        assert result['scope']=='ENGINEERING_ONLY_NOT_SIX_STATES'
        assert result['checks']==summary['checks'] and result['q']==result['p0_after']
        assert result['guidance']['status']=='accepted' and result['guidance']['loss_after_fixed']<result['guidance']['loss_before']
    else:
        assert not artifact.exists()
        if (out/'summary.json').exists():assert json.loads((out/'summary.json').read_text())['status']!='complete'
    record_property('scenario',scenario);record_property('outer_exit_code',code)
    record_property('controller_exit_code',receipt['exit_code']);record_property('status',receipt['status'])
    record_property('reason',receipt['reason']);record_property('owned_group_gone',receipt['cleanup']['owned_group_gone'])
    record_property('artifact_present',artifact.exists());record_property('wall_seconds',time.monotonic()-start)


def current_binding():
    return runner.stage_binding('1'*64,'2'*64,dict(python='engineering',numpy='engineering',scipy='engineering'))


def test_phase_binding_accepts_exact_match_and_rejects_old_receipt():
    binding=current_binding();receipt=dict(status='complete',scope='b1',**binding)
    assert len(binding['release_checklist_sha256'])==64
    runner.require_b1_binding(receipt,binding)
    del receipt['release_checklist_sha256']
    with pytest.raises(ValueError,match='release_checklist_sha256'):runner.require_b1_binding(receipt,binding)


@pytest.mark.parametrize('field',['release_checklist_sha256','implementation_sha256','target_sha256','portable_sha256','numeric_runtime'])
def test_phase_binding_rejects_each_changed_component(field):
    binding=current_binding();receipt=dict(status='complete',scope='b1',**copy.deepcopy(binding))
    receipt[field]={'scipy':'changed'} if field=='numeric_runtime' else 'f'*64
    with pytest.raises(ValueError,match=field):runner.require_b1_binding(receipt,binding)


@pytest.mark.parametrize('reused_source',['hsgd/kernel.py','stepwise_role/sampling.py','multiscale_role/evaluation.py'])
def test_reused_source_change_keeps_module_hash_but_changes_release_binding(reused_source,tmp_path,monkeypatch):
    # Copy real released source bytes into an ENGINEERING package, not fake
    # notebook manifests. A comment mutation stands for a different release.
    package=tmp_path/'engineering_package';module=package/'dynamic_role';module.mkdir(parents=True)
    for src in (ROOT/'dynamic_role').glob('*.py'):shutil.copyfile(src,module/src.name)
    relevant=['hsgd/kernel.py','stepwise_role/sampling.py','multiscale_role/evaluation.py']
    for name in relevant:
        dest=package/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
    def seal():
        names=sorted([str(p.relative_to(package)) for p in module.glob('*.py')]+relevant)
        (package/'DYNAMIC_SHA256SUMS').write_text(''.join(hashlib.sha256((package/n).read_bytes()).hexdigest()+'  '+n+'\n' for n in names))
    seal();monkeypatch.setattr(runner,'__file__',str(module/'runner.py'))
    before=current_binding();receipt=dict(status='complete',scope='b1',**before)
    with (package/reused_source).open('a') as f:f.write('\n# ENGINEERING different-release fixture only\n')
    seal();after=current_binding()
    assert before['implementation_sha256']==after['implementation_sha256']
    assert before['release_checklist_sha256']!=after['release_checklist_sha256']
    with pytest.raises(ValueError,match='release_checklist_sha256'):runner.require_b1_binding(receipt,after)


def test_actual_release_checklist_covers_reused_sources():
    files={line.split('  ',1)[1] for line in (ROOT/'DYNAMIC_SHA256SUMS').read_text().splitlines()}
    assert {'hsgd/kernel.py','hsgd/runtime.py','stepwise_role/sampling.py',
            'multiscale_role/evaluation.py','dynamic_role/core.py','run_dynamic.py'}<=files
