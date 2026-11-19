"""Finish fresh augmentation banks, fit three paired models, then frozen new-start suffixes."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];SRC=Path(__file__).parent
BASE=ROOT/'results/typed_edit_autonomous_20261118/cycle_repair_confirmation';OUT=BASE/'followthrough'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(n,v):(OUT/n).write_text(json.dumps(v,indent=2)+'\n')
def main():
    OUT.mkdir(parents=True,exist_ok=False)
    names=['cycle_confirmation_cache.py','cycle_confirmation_fit.py','cycle_confirmation_prefix_gpu.py',
           'cycle_confirmation_deploy.py','CYCLE_REPAIR_CONFIRMATION.md','CYCLE_REPAIR_GPU_PREFIX.md',Path(__file__).name]
    frozen={str(SRC/n):sha(SRC/n) for n in names};put('manifest.json',dict(files=frozen,new_coefficient_fits_max=3,new_original_MLP_fits=0,new_suffix_chains_max=24))
    for repeat in [2,3]:
        parent=BASE/f'cache/repeat{repeat}'
        assert read(parent/'acceptance.json')['passed'] and read(parent/'summary.json')['passed']
        for p,h in read(parent/'manifest.json')['files'].items():assert sha(Path(p))==h,p
        for p,h in read(parent/'acceptance.json')['outputs'].items():assert sha(Path(p))==h,p
    env=dict(os.environ,TMPDIR=str(ROOT/'.runtime_tmp'),TMP=str(ROOT/'.runtime_tmp'),TEMP=str(ROOT/'.runtime_tmp'),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    def run(name,args,destination,limit):
        for p,h in frozen.items():assert sha(Path(p))==h,p
        assert not destination.exists();tag=name+'_'+('_'.join(map(str,args)) or 'all')
        print('Coverage confirmation starting',tag,flush=True)
        with (OUT/f'{tag}.log').open('w') as log:
            child=subprocess.Popen([sys.executable,'-u',str(SRC/name),*map(str,args)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            put(f'{tag}_launch.json',dict(pid=child.pid,source_sha256=frozen[str(SRC/name)]))
            try:code=child.wait(timeout=limit)
            except subprocess.TimeoutExpired:
                child.terminate()
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:child.kill();child.wait()
                raise
        put(f'{tag}_exit.json',dict(returncode=code));assert code==0,tag
        assert read(destination/'acceptance.json')['passed']
        if name!='cycle_confirmation_deploy.py':assert read(destination/'summary.json')['passed']
    start=time.monotonic()
    run('cycle_confirmation_cache.py',[4],BASE/'cache/repeat4',900)
    for repeat in [1,2,3]:run('cycle_confirmation_fit.py',[repeat],BASE/f'fit/repeat{repeat}',1500)
    prefix=BASE/'prefix_gpu'
    while not (prefix/'acceptance.json').exists():
        assert not (prefix/'failure.json').exists(),'GPU prefix failed its gate or acceptance'
        assert time.monotonic()-start<5400
        time.sleep(5)
    assert read(prefix/'acceptance.json')['passed']
    run('cycle_confirmation_deploy.py',[],BASE/'deploy',9000)
    result=read(BASE/'deploy/summary.json')
    put('completion.json',dict(completed=True,core_gate_passed=result['passed'],Q_wins=sum(v.get('better_than_Q',False) for v in result['decisions']),seconds=time.monotonic()-start))
if __name__=='__main__':
    try:main()
    except BaseException as exc:
        if OUT.exists():put('failure.json',dict(type=type(exc).__name__,message=str(exc)))
        raise
