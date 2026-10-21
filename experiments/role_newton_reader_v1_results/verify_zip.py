import hashlib,json,os,subprocess,sys,tempfile,time,zipfile
from pathlib import Path
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    tick=time.perf_counter();r=json.loads((ROOT/'release_receipt.json').read_text());p=ROOT/'role_newton_reader_v1_review.zip';assert sha(p)==r['ZIP_sha256']
    d=Path(tempfile.mkdtemp(prefix='role-newton-review-'))
    with zipfile.ZipFile(p) as z:
        assert z.testzip() is None and not any(n.endswith('.pt') for n in z.namelist());z.extractall(d)
    root=d/'role_newton_reader_v1_review';lines=(root/'SHA256SUMS').read_text().splitlines()
    for line in lines:
        h,n=line.split('  ',1);assert sha(root/n)==h,n
    assert len(lines)==r['payload_files']
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',PYTEST_ADDOPTS='')
    subprocess.run([sys.executable,str(root/'experiments/role_newton_reader_v1/run.py'),'--help'],env=env,cwd=root,check=True,capture_output=True,timeout=10)
    tests=['role_newton_reader_v1/test_newton.py','role_newton_reader_v1/test_receipt.py','role_newton_reader_v1/test_imports.py',
           'role_logscale_reader_v1/test_scaling.py','role_logscale_reader_v1/test_additional.py','role_logscale_reader_v1/test_numeric_guard.py']
    cmd=['/usr/bin/time','-v',sys.executable,'-m','pytest','-q',*[str(root/'experiments'/n) for n in tests],'--junitxml='+str(ROOT/'unpacked_tests.xml')]
    with (ROOT/'unpacked_tests.log').open('w') as f:p=subprocess.run(cmd,env=env,cwd=root,stdout=f,stderr=subprocess.STDOUT,timeout=90)
    assert p.returncode==0
    new=ET.parse(ROOT/'unpacked_tests.xml');old=ET.parse(root/'results/tests_delivery.xml')
    def names(x):return sorted((a.get('classname').split('.')[-1],a.get('name')) for a in x.findall('.//testcase'))
    assert names(new)==names(old) and len(names(new))==35
    assert not new.findall('.//failure')+new.findall('.//error')+new.findall('.//skipped')
    result=dict(status='passed',payload_files=len(lines),tests=35,zero_skips=True,exact_test_set=True,entry_help=True,
        ZIP_sha256=r['ZIP_sha256'],private_weights_needed=False,NN_calls=0,chains=0,training=0,seconds=time.perf_counter()-tick,unpacked=str(root))
    (ROOT/'unpacked_acceptance.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
