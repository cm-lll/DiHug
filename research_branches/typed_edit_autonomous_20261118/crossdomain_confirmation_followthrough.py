"""Wait only for frozen independent banks and new prefixes, then evaluate all arms."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];SRC=Path(__file__).parent;BASE=ROOT/'results/typed_edit_crossdomain_augmentation_20261118'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(ds):
    out=BASE/f'confirmation/followthrough/{ds}';out.mkdir(parents=True,exist_ok=False)
    def put(n,v):(out/n).write_text(json.dumps(v,indent=2)+'\n')
    names=['crossdomain_confirmation_followthrough.py','crossdomain_confirmation_deploy.py','CROSSDOMAIN_AUGMENTATION_CONFIRMATION.md']
    frozen={str(SRC/n):sha(SRC/n) for n in names};put('manifest.json',dict(files=frozen,no_parameter_search=True))
    banks=BASE/f'confirmation/bank_followthrough/{ds}';prefix=BASE/f'confirmation/prefix/{ds}';start=time.monotonic()
    while not ((banks/'completion.json').exists() and (prefix/'acceptance.json').exists()):
        assert time.monotonic()-start<14400
        if (prefix/'failure.json').exists():put('completion.json',dict(activated=False,reason='prefix implementation or cost gate failed'));return
        if (banks/'completion.json').exists() and not read(banks/'completion.json').get('passed',False):put('completion.json',dict(activated=False,reason='bank/fit gate failed'));return
        time.sleep(5)
    assert read(banks/'completion.json')['passed'] and read(prefix/'acceptance.json')['passed']
    for p,h in frozen.items():assert sha(p)==h,p
    put('activation.json',dict(bank_completion_sha256=sha(banks/'completion.json'),prefix_acceptance_sha256=sha(prefix/'acceptance.json')))
    env=dict(os.environ,TMPDIR=str(ROOT/'.runtime_tmp'),TMP=str(ROOT/'.runtime_tmp'),TEMP=str(ROOT/'.runtime_tmp'),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    with (out/'deploy.log').open('w') as log:res=subprocess.run([sys.executable,'-u',str(SRC/'crossdomain_confirmation_deploy.py'),ds],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=8700)
    target=BASE/f'confirmation/deploy/{ds}';accepted=res.returncode==0 and (target/'acceptance.json').exists() and read(target/'acceptance.json')['passed']
    put('completion.json',dict(activated=True,returncode=res.returncode,implementation_accepted=accepted,scientific_gate_passed=read(target/'summary.json')['passed'] if accepted else None,seconds=time.monotonic()-start));print(ds,'confirmation followthrough finished',accepted,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset',choices=['pubmed','acm']);a=p.parse_args();main(a.dataset)
