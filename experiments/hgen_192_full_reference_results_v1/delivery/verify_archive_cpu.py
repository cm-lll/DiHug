"""Independent extraction + portable CPU recomputation of all delivered graphs."""
import pathlib,json,hashlib,zipfile,subprocess,os,time,math
ROOT=pathlib.Path(__file__).resolve().parent
PYTHON='/data2/lyh/miniconda3/envs/hgen_env/bin/python'
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def main():
 begin=time.time();receipt=json.loads((ROOT/'package_receipt.json').read_text());archive=pathlib.Path(receipt['zip']);assert sha(archive)==receipt['sha256'];target=ROOT/'tmp/independent_unpack';target.mkdir(exist_ok=False)
 with zipfile.ZipFile(archive) as z:
  for name in z.namelist():
   assert not pathlib.PurePosixPath(name).is_absolute() and '..' not in pathlib.PurePosixPath(name).parts
  z.extractall(target)
 manifest=json.loads((target/'payload_manifest.json').read_text());checks=0
 for name,v in manifest.items():
  p=target/name;assert p.stat().st_size==v['bytes'] and sha(p)==v['sha256'];checks+=1
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TMPDIR=str(ROOT/'tmp'),PYTHONDONTWRITEBYTECODE='1')
 with (ROOT/'independent_recompute.log').open('w') as f:code=subprocess.call([PYTHON,'-B',str(target/'recompute_cpu.py'),'--root',str(target),'--output',str(target/'new_cpu_recompute')],env=env,stdout=f,stderr=subprocess.STDOUT)
 assert code==0,'portable CPU recomputation failed'
 doc=json.loads((target/'evaluation.json').read_text());exact=0;tolerant=0
 for r in doc['records']:
  key=f"{r['dataset']}_s{r['train_seed']}_{r['sample_seed']}_{r['version']}";new=json.loads((target/'new_cpu_recompute'/(key+'.json')).read_text())
  for group in ('raw','reference','absolute_error','distribution_error'):
   assert set(new[group])==set(r[group])
   for k,a in r[group].items():
    b=new[group][k]
    if a==b:exact+=1
    else:
     assert a is not None and b is not None and math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-10),(key,group,k,a,b);tolerant+=1
 result=dict(status='passed',payload_hash_checks=checks,ZIP_SHA256=receipt['sha256'],graph_versions_recomputed=len(doc['records']),scalar_exact=exact,scalar_tolerance_only=tolerant,predeclared_FP64_relative_tolerance=1e-9,predeclared_FP64_absolute_tolerance=1e-10,model_forwards=0,training_updates=0,new_graphs=0,seconds=time.time()-begin,epoch=time.time(),extracted_directory=str(target))
 (ROOT/'archive_audit.json').write_text(json.dumps(result,indent=2));print(result)
if __name__=='__main__':main()
