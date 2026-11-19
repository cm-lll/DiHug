"""One reference's gated fresh-bank/fresh-start confirmation; bounded child phases."""
import argparse,concurrent.futures,hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];SRC=ROOT/'research_branches/typed_edit_autonomous_20261118'
BASE=ROOT/'results/typed_edit_autonomous_20261118';OUT=None
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(n,v):(OUT/n).write_text(json.dumps(v,indent=2)+'\n')
def main(ds):
    global OUT
    OUT=BASE/f'additional_confirmation_v1/followthrough/{ds}';OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    names=['ADDITIONAL_CONFIRMATION_V1.md','additional_confirmation_followthrough.py','additional_confirmation_bank.py','additional_confirmation_cache.py','additional_confirmation_fit.py','additional_confirmation_prefix.py','additional_confirmation_deploy.py','acm_scaled_solver.py','acm_guidance4_adapter.py','freebase_legal_adapter.py','freebase_legal_replay.py','prefix_replay.py']
    frozen={str(SRC/n):sha(SRC/n) for n in names};put('manifest.json',dict(files=frozen,dataset=ds,new_coefficient_fits_maximum=3,new_original_MLP_fits=0,corruption_repeats=[2,3,4],starts=[15001,15002],maximum_concurrent_GPUs=1))
    pilot=BASE/({'acm':'acm_guidance4','freebase':'freebase_legal_selection'}[ds])/'deploy'
    parent=pilot.parent/'followthrough'
    while not (pilot/'acceptance.json').exists():
        assert time.monotonic()-start<21600
        if (pilot/'failure.json').exists() or (parent/'completion.json').exists():
            put('completion.json',dict(activated=False,reason='pilot did not produce an accepted complete suffix panel'));return
        time.sleep(5)
    if not read(pilot/'acceptance.json')['passed'] or not read(pilot/'summary.json')['passed']:
        put('completion.json',dict(activated=False,reason='pilot failed fixed core gate'));return
    put('activation.json',dict(pilot_acceptance_sha256=sha(pilot/'acceptance.json'),pilot_summary_sha256=sha(pilot/'summary.json')))
    env=dict(os.environ,TMPDIR=str(ROOT/'.runtime_tmp'),TMP=str(ROOT/'.runtime_tmp'),TEMP=str(ROOT/'.runtime_tmp'),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    def run(phase,rep=None):
        for p,h in frozen.items():assert sha(p)==h,p
        source='additional_confirmation_'+phase+'.py';key=phase+('' if rep is None else '_'+str(rep))
        result=BASE/f'additional_confirmation_v1/{phase}/{ds}'
        if rep is not None:result=result/str(rep)
        assert not result.exists()
        args=[sys.executable,'-u',str(SRC/source),ds]+([] if rep is None else [str(rep)])
        with (OUT/(key+'.log')).open('w') as log:
            proc=subprocess.Popen(args,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            put(key+'.launch.json',dict(pid=proc.pid,source_sha256=frozen[str(SRC/source)],started_monotonic=time.monotonic()))
            limit={'cache':10500,'fit':1500,'prefix':6900,'deploy':9000}[phase]
            try:code=proc.wait(timeout=limit)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:proc.wait(timeout=15)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
                put(key+'.exit.json',dict(accepted=False,reason='child time bound',limit=limit));return False
        ok=code==0 and (result/'acceptance.json').exists() and read(result/'acceptance.json')['passed']
        if phase=='fit' and ok:ok=read(result/'summary.json')['passed']
        put(key+'.exit.json',dict(exit_code=code,accepted=ok));print('Additional confirmation',ds,key,ok,flush=True);return ok
    def banks():
        outcomes={}
        for rep in [2,3,4]:outcomes[str(rep)]=run('cache',rep) and run('fit',rep)
        return outcomes
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future=pool.submit(banks);prefix_ok=run('prefix');fits=future.result()
    deploy_ok=run('deploy') if prefix_ok and all(fits.values()) else False
    science=read(BASE/f'additional_confirmation_v1/deploy/{ds}/summary.json')['passed'] if deploy_ok else False
    put('completion.json',dict(activated=True,dataset=ds,fits=fits,prefix_accepted=prefix_ok,deploy_accepted=deploy_ok,scientific_gate_passed=science,seconds=time.monotonic()-start))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset',choices=['acm','freebase']);a=p.parse_args()
    try:main(a.dataset)
    except BaseException as exc:
        if OUT is not None:put('failure.json',dict(type=type(exc).__name__,message=str(exc)))
        raise
