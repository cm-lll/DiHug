"""Independent extraction, all hashes, exact33-test set, no scientific rerun."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
import zipfile

ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    started=time.monotonic();r=read(ROOT/'release_receipt.json')
    zipfile_path=ROOT/'role_unseen_target_validation_v1_review.zip'
    assert sha(zipfile_path)==r['sha256']
    d=Path(tempfile.mkdtemp(prefix='role-unseen-review-'))
    with zipfile.ZipFile(zipfile_path) as z:
        assert z.testzip() is None
        assert not any(n.endswith('.pt') for n in z.namelist())
        z.extractall(d)
    bundle=d/'role_unseen_target_validation_v1_review';n=0
    for line in (bundle/'SHA256SUMS').read_text().splitlines():
        h,p=line.split('  ',1);assert sha(bundle/p)==h,p;n+=1
    assert n==r['payload_files']==507
    evidence=bundle/'results'
    for p,h in read(evidence/'run01/source_manifest.json').items():
        assert sha(bundle/'experiments'/p)==h,p
    assert len(list((evidence/'run01').glob('seed*/*_result.json')))==120
    assert read(evidence/'run01/controller.json')['status']=='failed'
    assert read(evidence/'audit_repair_v1/receipt.json')['status']=='completed'
    assert read(evidence/'audit_repair_v1/audit.json')['graphs']==120
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        PYTEST_ADDOPTS='',ROLE_REVIEW_SOURCE_ROOT=str(bundle),PYTHONPATH=str(evidence))
    helpcmd=[sys.executable,str(bundle/'experiments/role_unseen_target_validation_v1/execute.py'),'--help']
    subprocess.run(helpcmd,cwd=bundle,env=env,check=True,stdout=subprocess.PIPE,timeout=20)
    tests=[str(bundle/'experiments/role_unseen_target_validation_v1/test_execution.py'),
           str(bundle/'experiments/role_target_scalability_v1/test_planning.py'),str(evidence/'test_repair.py')]
    cmd=['/usr/bin/time','-v',sys.executable,'-m','pytest','-q',*tests,'--import-mode=importlib',
         '--junitxml='+str(ROOT/'unpacked_tests.xml')]
    with (ROOT/'unpacked_tests.log').open('w') as f:
        done=subprocess.run(cmd,cwd=bundle,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=120)
    assert done.returncode==0
    new=ET.parse(ROOT/'unpacked_tests.xml')
    def ids(x):return sorted((c.get('classname').split('.')[-1],c.get('name')) for c in x.findall('.//testcase'))
    wanted=ids(ET.parse(evidence/'preflight_tests.xml'))+ids(ET.parse(evidence/'repair_tests_complete.xml'))
    assert ids(new)==sorted(wanted) and len(wanted)==33
    assert not new.findall('.//failure')+new.findall('.//error')+new.findall('.//skipped')
    result=dict(status='passed',payload_files=n,exact_test_set=True,tests=33,failures=0,errors=0,skips=0,
        frozen_source_hashes_match=True,entry_help='passed',old_failure_preserved=True,
        scientific_chains_rerun=0,training_updates=0,GPU_used=False,wall_seconds=time.monotonic()-started,
        zip_sha256=r['sha256'],unpacked_directory=str(bundle))
    (ROOT/'unpacked_acceptance.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
