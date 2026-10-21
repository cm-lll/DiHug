"""Fresh extraction/hash checks and identical33-test collection; never science."""
import hashlib,json,os,subprocess,sys,tempfile,time,zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    start=time.monotonic();receipt=read(ROOT/'release_receipt.json');archive=ROOT/'role_condition_coverage_v1_review.zip'
    assert sha(archive)==receipt['sha256']
    d=Path(tempfile.mkdtemp(prefix='role-coverage-review-'))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and not any(n.endswith('.pt') for n in z.namelist())
        z.extractall(d)
    bundle=d/'role_condition_coverage_v1_review';count=0
    for line in (bundle/'SHA256SUMS').read_text().splitlines():
        h,n=line.split('  ',1);assert sha(bundle/n)==h,n;count+=1
    assert count==receipt['payload_files']
    frozen=read(bundle/'results/freeze.json')
    for n,h in frozen['source'].items():assert sha(bundle/'experiments'/n)==h
    for n,h in frozen['inputs'].items():assert sha(bundle/'results'/n)==h
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTEST_ADDOPTS='')
    r=subprocess.run([sys.executable,str(bundle/'experiments/role_condition_coverage_v1/run.py'),'--help'],env=env,cwd=bundle,capture_output=True,timeout=20)
    assert r.returncode==0
    # Test-only relocate the frozen results root BEFORE dependent modules import it.
    # Original initialization weights are explicitly local, read-only and not shipped.
    bootstrap=d/'test_bootstrap.py'
    bootstrap.write_text("import sys\nfrom pathlib import Path\n"+
        'sys.path.insert(0,'+repr(str(bundle/'experiments/role_condition_coverage_v1'))+')\n'+
        'import common\ncommon.RESULT=Path('+repr(str(bundle/'results'))+')\n'+
        'import pytest\nsys.exit(pytest.main(sys.argv[1:]))\n')
    tests=[bundle/'experiments/role_condition_coverage_v1/test_coverage.py',
           bundle/'experiments/role_specialization_v1/test_capacity.py',
           bundle/'experiments/role_robust_training_v1/test_6000.py']
    cmd=['/usr/bin/time','-v',sys.executable,str(bootstrap),'-q',*map(str,tests),'--import-mode=importlib',
         '--junitxml='+str(ROOT/'unpacked_tests.xml')]
    with (ROOT/'unpacked_tests.log').open('w') as f:r=subprocess.run(cmd,env=env,cwd=bundle,stdout=f,stderr=subprocess.STDOUT,timeout=120)
    assert r.returncode==0
    old=ET.parse(bundle/'results/tests.xml');new=ET.parse(ROOT/'unpacked_tests.xml')
    def ids(x):return sorted((c.get('classname').split('.')[-1],c.get('name')) for c in x.findall('.//testcase'))
    assert ids(old)==ids(new) and len(ids(new))==33
    assert not new.findall('.//failure')+new.findall('.//error')+new.findall('.//skipped')
    result=dict(status='passed',payload_files=count,exact_test_set=True,tests=33,failures=0,errors=0,skips=0,
        source_and_input_hashes_match=True,entry_help='passed',unpacked_directory=str(bundle),
        science_updates=0,science_chains=0,GPU_used=False,wall_seconds=time.monotonic()-start,ZIP_sha256=receipt['sha256'])
    (ROOT/'unpacked_acceptance.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
