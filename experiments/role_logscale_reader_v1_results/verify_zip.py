"""Fresh extraction, hashes, entry help and exact21-test set; no private weights."""
import hashlib,json,os,subprocess,sys,tempfile,time,zipfile
from pathlib import Path
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    tick=time.perf_counter();r=json.loads((ROOT/'release_receipt.json').read_text());zpath=ROOT/'role_logscale_reader_v1_review.zip'
    assert sha(zpath)==r['ZIP_sha256'];d=Path(tempfile.mkdtemp(prefix='role-logscale-unpack-'))
    with zipfile.ZipFile(zpath) as z:
        assert z.testzip() is None and not any(n.endswith('.pt') for n in z.namelist());z.extractall(d)
    root=d/'role_logscale_reader_v1_review';lines=(root/'SHA256SUMS').read_text().splitlines()
    for line in lines:
        h,n=line.split('  ',1);assert sha(root/n)==h,n
    assert len(lines)==r['payload_files']
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',PYTEST_ADDOPTS='')
    subprocess.run([sys.executable,str(root/'experiments/role_logscale_reader_v1/run.py'),'--help'],cwd=root,env=env,check=True,capture_output=True,timeout=10)
    tests=[str(root/'experiments/role_logscale_reader_v1'/n) for n in ('test_scaling.py','test_additional.py','test_numeric_guard.py')]
    with (ROOT/'unpacked_tests.log').open('w') as f:
        p=subprocess.run(['/usr/bin/time','-v',sys.executable,'-m','pytest','-q',*tests,'--junitxml='+str(ROOT/'unpacked_tests.xml')],cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=60)
    assert p.returncode==0
    new=ET.parse(ROOT/'unpacked_tests.xml');old=ET.parse(ROOT/'evidence/tests_delivery.xml')
    def names(x):return sorted((p.get('classname').split('.')[-1],p.get('name')) for p in x.findall('.//testcase'))
    assert names(new)==names(old) and len(names(new))==21
    assert not new.findall('.//error')+new.findall('.//failure')+new.findall('.//skipped')
    receipt=dict(status='passed',payload_files=len(lines),ZIP_sha256=r['ZIP_sha256'],tests=21,zero_skips=True,exact_set=True,
        no_private_weights_needed=True,entry_help=True,training_updates=0,NN_diagnostic_calls=0,science_chains=0,
        seconds=time.perf_counter()-tick,unpacked_path=str(root))
    (ROOT/'unpacked_acceptance.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
if __name__=='__main__':main()
