"""Independent extraction, all hashes/CLI/exact test collection and zero skips."""
import hashlib,json,os,subprocess,sys,tempfile,time,zipfile,xml.etree.ElementTree as ET
from pathlib import Path
R=Path(__file__).resolve().parent
def main():
    started=time.perf_counter();archive=R/'dihug_imdb_role_pair_gpu_v1.zip';receipt=json.loads((R/'release_receipt.json').read_text());assert hashlib.sha256(archive.read_bytes()).hexdigest()==receipt['archive_sha256']
    temp=Path(tempfile.mkdtemp(prefix='imdb-role-pair-review-',dir='/data2/lyh/gnn_project/DiHug/results'))
    with zipfile.ZipFile(archive) as z:
        for n in z.namelist():assert not Path(n).is_absolute() and '..' not in Path(n).parts
        z.extractall(temp)
    root=temp/'imdb_role_pair_gpu_v1';count=0
    for line in (root/'SHA256SUMS').read_text().splitlines():
        h,n=line.split('  ',1);assert hashlib.sha256((root/n).read_bytes()).hexdigest()==h;count+=1
    p=root/'package';env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTEST_ADDOPTS='',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',PYTHONPATH=str(p))
    subprocess.run([sys.executable,str(p/'run.py'),'--help'],check=True,capture_output=True,env=env,cwd=p,timeout=10)
    subprocess.run([sys.executable,str(p/'run.py'),'plan'],check=True,capture_output=True,env=env,cwd=p,timeout=10)
    with (R/'unpacked_tests.log').open('w') as log:
        result=subprocess.run([sys.executable,'-m','pytest','-q','-c','/dev/null','--confcutdir='+str(p),str(p/'tests'),'--junitxml='+str(R/'unpacked_tests.xml')],cwd=p,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120)
    assert result.returncode==0
    x=ET.parse(R/'unpacked_tests.xml');expected=ET.parse(R/'evidence/tests_final.xml')
    names=lambda t:sorted((e.get('classname').split('.')[-1],e.get('name')) for e in t.findall('.//testcase'))
    assert names(x)==names(expected) and not x.findall('.//failure')+x.findall('.//error')+x.findall('.//skipped')
    record={'status':'passed','checksummed_files':count,'test_count':len(names(x)),'exact_test_collection':True,'zero_skips':True,'entry_help_and_plan':True,
        'wall_seconds':time.perf_counter()-started,'source':str(root),'notebook_or_GPU_test':False,'formal_training_or_large_chain':False}
    (R/'unpacked_acceptance.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
if __name__=='__main__':main()
