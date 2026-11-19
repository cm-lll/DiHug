"""At most two independent cache producers, after the accepted pilot gate."""
import argparse,concurrent.futures,hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];SRC=Path(__file__).parent;BASE=ROOT/'results/typed_edit_crossdomain_augmentation_20261118'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(ds):
    out=BASE/f'confirmation/bank_followthrough/{ds}';out.mkdir(parents=True,exist_ok=False)
    def put(n,v):(out/n).write_text(json.dumps(v,indent=2)+'\n')
    names=['crossdomain_confirmation_banks.py','crossdomain_confirmation_cache.py','crossdomain_confirmation_fit.py','crossdomain_augmentation_cache.py','crossdomain_augmentation_finalize_v2.py','CROSSDOMAIN_AUGMENTATION_CONFIRMATION.md']
    frozen={str(SRC/n):sha(SRC/n) for n in names};put('manifest.json',dict(files=frozen,maximum_cache_workers=2,maximum_new_coefficient_fits=6,new_original_MLP_fits=0))
    pilot=BASE/f'deploy/{ds}/bank1';start=time.monotonic()
    while not (pilot/'acceptance.json').exists():
        assert time.monotonic()-start<4000
        if (pilot/'failure.json').exists():put('completion.json',dict(activated=False,reason='pilot failed implementation acceptance'));return
        time.sleep(5)
    assert read(pilot/'acceptance.json')['passed']
    if not read(pilot/'summary.json')['passed']:put('completion.json',dict(activated=False,reason='pilot scientific gate failed'));return
    put('activation.json',dict(pilot_acceptance_sha256=sha(pilot/'acceptance.json'),pilot_summary_sha256=sha(pilot/'summary.json')))
    env=dict(os.environ,TMPDIR=str(ROOT/'.runtime_tmp'),TMP=str(ROOT/'.runtime_tmp'),TEMP=str(ROOT/'.runtime_tmp'),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    def bank_work(bank):
        repeat=(bank-1) if ds=='pubmed' else bank;phases={}
        for phase,extra,bound,target in [('cache',['--bank',str(bank)],3700,BASE/f'cache/{ds}/bank{bank}'),('fit',['--repeat',str(repeat),'--bank',str(bank)],2500,BASE/f'confirmation/fit/{ds}/bank{bank}')]:
            for p,h in frozen.items():assert sha(p)==h,p
            with (out/f'bank{bank}_{phase}.log').open('w') as log:res=subprocess.run([sys.executable,'-u',str(SRC/f'crossdomain_confirmation_{phase}.py'),ds,*extra],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=bound)
            ok=res.returncode==0 and (target/'acceptance.json').exists() and read(target/'acceptance.json')['passed'];phases[phase]=ok;put(f'bank{bank}_{phase}.exit.json',dict(returncode=res.returncode,accepted=ok))
            if not ok or not read(target/'summary.json')['passed']:break
        return dict(bank=bank,repeat=repeat,phases=phases)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:rows=list(pool.map(bank_work,[2,3,4]))
    passed=all(r['phases'].get('cache') and r['phases'].get('fit') for r in rows)
    put('completion.json',dict(activated=True,completed=True,passed=passed,rows=rows,seconds=time.monotonic()-start));print(ds,'fresh banks complete',passed,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset',choices=['pubmed','acm']);a=p.parse_args();main(a.dataset)
