"""Run only the registered F fit and paired suffix after accepted dependencies."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];SRC=ROOT/'research_branches/typed_edit_autonomous_20261118';OUT=ROOT/'results/typed_edit_autonomous_20261118/confirmation_v1/followthrough'
BASE=OUT.parent

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(name,x):(OUT/name).write_text(json.dumps(x,indent=2)+'\n')

def main():
    OUT.mkdir(parents=True,exist_ok=False)
    files=[Path(__file__),SRC/'confirmation_fit.py',SRC/'confirmation_deploy.py',SRC/'CONFIRMATION_V1.md'];frozen={str(p):sha(p) for p in files}
    put('manifest.json',dict(files=frozen,phases=['fit','deploy'],new_original_MLP_training=False));begin=time.monotonic()
    for phase,parents in [('fit',['cache_accounted_v2']),('deploy',['cache_accounted_v2','fit','prefix_v2'])]:
        last=0
        while True:
            assert time.monotonic()-begin<21600,'dependency wait exceeded six hours'
            for parent in parents:
                assert not (BASE/parent/'failure.json').exists(),f'dependency failed: {parent}'
            ready=all((BASE/p/'acceptance.json').exists() and read(BASE/p/'acceptance.json')['passed'] for p in parents)
            if ready:break
            if time.monotonic()-last>=60:
                print('confirmation pending',phase,parents,round(time.monotonic()-begin,1),flush=True);last=time.monotonic()
            time.sleep(5)
        if phase=='deploy':assert read(BASE/'fit/summary.json')['passed'],'coefficient numerical gate failed'
        for name,h in frozen.items():assert sha(Path(name))==h,name
        assert not (BASE/phase).exists(),f'will not overwrite {phase}'
        env=dict(os.environ,TMPDIR=str(ROOT/'.runtime_tmp'),TMP=str(ROOT/'.runtime_tmp'),TEMP=str(ROOT/'.runtime_tmp'),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        print('confirmation starting',phase,flush=True)
        with (OUT/f'{phase}.log').open('w') as log:
            child=subprocess.Popen([sys.executable,'-u',str(SRC/f'confirmation_{phase}.py')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            put(f'{phase}_launch.json',dict(pid=child.pid,source_sha256=frozen[str(SRC/f'confirmation_{phase}.py')],started_monotonic=time.monotonic()))
            while child.poll() is None:
                assert time.monotonic()-begin<21600,'total supervisor bound exceeded'
                time.sleep(5)
            assert child.returncode==0,f'{phase} failed; inspect {OUT/phase}.log'
        assert read(BASE/phase/'acceptance.json')['passed'];print('confirmation accepted',phase,flush=True)
    put('completion.json',dict(completed=True,core_gate_passed=read(BASE/'deploy/summary.json')['passed'],seconds=time.monotonic()-begin))

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        if OUT.exists():put('failure.json',dict(type=type(exc).__name__,message=str(exc)))
        raise
