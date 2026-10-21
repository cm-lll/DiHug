"""Temporary engineering checkpoints only; no scientific training."""
import copy,json,os,random,subprocess,sys
from pathlib import Path
import numpy as np
import pytest
import torch
import pair_core as p

def value(u):
    return {'binding':{'fixture':'pointer regression'},'updates':u,'model':torch.tensor([float(u)])}

def test_recovery_pointer_second_resume_then_stop_before_milestone(tmp_path):
    p.checkpoint(tmp_path,value(0))
    p.rolling_recovery(tmp_path,value(123))
    assert p.load_checkpoint(tmp_path,value(0)['binding'])['updates']==123
    assert p.load_checkpoint(tmp_path,value(0)['binding'])['updates']==123
    p.rolling_recovery(tmp_path,value(127))
    assert p.load_checkpoint(tmp_path,value(0)['binding'])['updates']==127
    assert p.load_checkpoint(tmp_path,value(0)['binding'])['updates']==127
    for u in (130,133,136):p.rolling_recovery(tmp_path,value(u))
    assert p.load_checkpoint(tmp_path,value(0)['binding'])['updates']==136
    assert len(list((tmp_path/'recovery').glob('*.pt')))==2

def test_permanent_pointer_repeat_restore_and_evaluation_file(tmp_path):
    for u in (0,1000):p.checkpoint(tmp_path,value(u))
    for _ in range(2):assert p.load_checkpoint(tmp_path,value(0)['binding'])['updates']==1000
    assert torch.load(tmp_path/'pair_u00001000.pt',weights_only=False)['updates']==1000

@pytest.mark.parametrize('mutation',['missing','hash','updates','payload','absolute','escape','symlink'])
def test_checkpoint_pointer_conflicts_fail_closed(tmp_path,mutation):
    folder=tmp_path/'checkpoints';folder.mkdir();file=p.checkpoint(folder,value(1000));ptr=p.read(folder/'latest.json')
    if mutation=='missing':file.unlink()
    elif mutation=='hash':ptr['sha256']='0'*64
    elif mutation=='updates':ptr['updates']=1001
    elif mutation=='payload':ptr['payload_digest']='0'*64
    elif mutation=='absolute':ptr['path']=str(file.resolve())
    elif mutation=='escape':ptr['path']='../'+file.name
    else:
        other=tmp_path/file.name;file.rename(other);file.symlink_to(other)
    p.save(folder/'latest.json',ptr)
    with pytest.raises((ValueError,FileNotFoundError)):p.load_checkpoint(folder,value(0)['binding'])

def test_same_update_two_locations_conflict_rejected(tmp_path):
    p.checkpoint(tmp_path,value(10));v=value(10);v['model']=torch.tensor([999.])
    p.rolling_recovery(tmp_path,v)
    with pytest.raises(ValueError,match='conflict'):p.load_checkpoint(tmp_path,v['binding'])

def test_two_arms_optimizers_rng_and_following_input_new_process(tmp_path):
    from restore_pair_worker import fresh,step,pack
    p.setup('cpu',True);torch.manual_seed(81);random.seed(82);np.random.seed(83);rng=np.random.default_rng(84)
    models,opts=fresh();folder=tmp_path/'checkpoints';folder.mkdir()
    p.checkpoint(folder,pack(models,opts,rng,0))
    step(models,opts,rng,0);p.rolling_recovery(folder,pack(models,opts,rng,1))
    ck=p.load_checkpoint(folder,{'fixture':'two-arm fresh-process'});assert ck['updates']==1
    step(models,opts,rng,1);saved=pack(models,opts,rng,2);p.rolling_recovery(folder,saved)
    assert p.load_checkpoint(folder,saved['binding'])['updates']==2
    inp=step(models,opts,rng,2);expected=pack(models,opts,rng,3)
    dest=tmp_path/'continued.pt';env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONPATH=str(p.ROOT))
    worker=Path(__file__).with_name('restore_pair_worker.py')
    proc=subprocess.run([sys.executable,str(worker),str(folder),str(dest)],env=env,capture_output=True,text=True,timeout=30)
    assert proc.returncode==0,proc.stdout+proc.stderr
    got=torch.load(dest,weights_only=False)
    assert got['restored']==p.digest(saved) and got['input']==inp and p.digest(got['after'])==p.digest(expected)
    assert all(int(state['step'])==3 for opt in got['after']['optimizers'].values() for state in opt['state'].values())

@pytest.mark.parametrize('bad',['arm','step'])
def test_paired_content_rejected_even_with_valid_recomputed_digest(tmp_path,bad):
    from restore_pair_worker import fresh,step,pack
    models,opts=fresh();rng=np.random.default_rng(84);step(models,opts,rng,0);v=pack(models,opts,rng,1)
    if bad=='arm':v['optimizers'].pop('role')
    else:next(iter(v['optimizers']['role']['state'].values()))['step']=torch.tensor(9.)
    p.checkpoint(tmp_path,v)
    with pytest.raises(ValueError,match='paired'):p.load_checkpoint(tmp_path,v['binding'])

def test_storage_fresh_and_resume_credit_real_retained_checkpoints(tmp_path):
    cfg=p.config();initial=p.storage_requirement(tmp_path,cfg,False)
    assert initial['required_free_bytes']>=16*1024**3 and initial['remaining_permanent_checkpoint_files']==39
    folder=tmp_path/f"seed{cfg['train_seeds'][0]}"/'checkpoints';folder.mkdir(parents=True)
    for u in (0,1000,2000):
        with (folder/f'pair_u{u:08d}.pt').open('wb') as f:f.truncate(160*1024**2)
    result=p.storage_requirement(tmp_path,cfg,True)
    assert result['retained_inventory_credit_bytes']==3*160*1024**2
    assert result['remaining_permanent_checkpoint_files']==36
    assert result['required_free_bytes']<initial['required_free_bytes']
    assert result['required_free_bytes']>2*1024**3 and not result['persistence_verified']
    assert result['filesystem_device_id']==tmp_path.stat().st_dev and result['output_path']==str(tmp_path.resolve())
    assert result['mount'] is None or Path(result['mount']['mount_point']) in (tmp_path.resolve(),*tmp_path.resolve().parents)

def test_space_gate_checks_output_volume_and_keeps_existing_files(tmp_path,monkeypatch):
    from types import SimpleNamespace
    seen=[];existing=tmp_path/'old-result.txt';existing.write_text('preserve')
    def low(path):seen.append(Path(path));return SimpleNamespace(free=1024)
    monkeypatch.setattr(p.shutil,'disk_usage',low)
    with pytest.raises(OSError,match='insufficient free space'):p.check_storage(tmp_path,p.config(),True)
    assert seen==[tmp_path.resolve()] and existing.read_text()=='preserve'
    assert not p.read(tmp_path/'storage.json')['sufficient']

def test_space_gate_preserves_safety_margin_after_all_milestones(tmp_path,monkeypatch):
    from types import SimpleNamespace
    cfg=p.config()
    for seed in cfg['train_seeds']:
        folder=tmp_path/f'seed{seed}'/'checkpoints';folder.mkdir(parents=True)
        for u in range(0,12001,1000):
            with (folder/f'pair_u{u:08d}.pt').open('wb') as f:f.truncate(160*1024**2)
    report=p.storage_requirement(tmp_path,cfg,True)
    assert report['remaining_permanent_checkpoint_files']==0 and 2*1024**3<report['required_free_bytes']<16*1024**3
    monkeypatch.setattr(p.shutil,'disk_usage',lambda path:SimpleNamespace(free=report['required_free_bytes']))
    assert p.check_storage(tmp_path,cfg,True)['sufficient']
    monkeypatch.setattr(p.shutil,'disk_usage',lambda path:SimpleNamespace(free=report['required_free_bytes']-1))
    with pytest.raises(OSError):p.check_storage(tmp_path,cfg,True)

def test_preflight_supervisor_timeout_is_incomplete_not_algorithm_failure(tmp_path):
    from run import supervise
    code=supervise([sys.executable,'-c','import time;time.sleep(20)'],tmp_path,__import__('time').time()+.1,'preflight.json',grace=.1)
    result=p.read(tmp_path/'preflight.json')
    assert code!=0 and result['status']=='incomplete' and result['not_algorithm_failure']
    assert not result['target_GPU_gate_passed'] and not result['full_IMDB_cost_complete']
