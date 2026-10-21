import json,hashlib,os,sys,subprocess,tempfile,time
from pathlib import Path
import xml.etree.ElementTree as ET
from materialize import materialize
ROOT=Path(__file__).resolve().parent
def main():
    tick=time.perf_counter();r=json.loads((ROOT/'release_receipt.json').read_text());a=ROOT/'role_newton_large_model_v1_review.zip';assert hashlib.sha256(a.read_bytes()).hexdigest()==r['ZIP_sha256']
    base=Path(tempfile.mkdtemp(prefix='large-probe-unpack-'))/'restored';count=materialize(a,base);assert count==r['logical_files']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTEST_ADDOPTS='')
    subprocess.run([sys.executable,str(base/'experiments/role_newton_large_model_v1/probe.py'),'--help'],cwd=base,env=env,check=True,capture_output=True,timeout=15)
    tests=['role_newton_large_model_v1/test_probe.py','role_newton_generation_v1/test_integration.py','role_newton_reader_v1/test_newton.py','role_newton_reader_v1/test_receipt.py','role_newton_reader_v1/test_imports.py','role_logscale_reader_v1/test_scaling.py','role_logscale_reader_v1/test_additional.py','role_logscale_reader_v1/test_numeric_guard.py']
    cmd=[sys.executable,'-m','pytest','-q',*[str(base/'experiments'/n) for n in tests],'--junitxml='+str(ROOT/'unpacked_tests.xml')]
    with (ROOT/'unpacked_tests.log').open('w') as f:p=subprocess.run(cmd,env=env,cwd=base,stdout=f,stderr=subprocess.STDOUT,timeout=60)
    assert p.returncode==0
    a=ET.parse(ROOT/'unpacked_tests.xml');b=ET.parse(base/'results/tests_final.xml')
    def names(x):return sorted((t.get('classname').split('.')[-1],t.get('name')) for t in x.findall('.//testcase'))
    assert names(a)==names(b) and len(names(a))==43 and not a.findall('.//failure')+a.findall('.//error')+a.findall('.//skipped')
    result=dict(status='passed',logical_files=count,tests=43,zero_skips=True,exact_test_set=True,entry_help=True,seconds=time.perf_counter()-tick,large_model_probes=0,training=0,GPU=0,fixture_small_single_steps=2)
    (ROOT/'unpacked_acceptance.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
