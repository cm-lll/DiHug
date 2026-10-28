"""Package existing evidence only; no scientific computation or model loading."""
import csv,hashlib,json,pathlib,shutil,zipfile,datetime,tempfile
ROOT=pathlib.Path('/data2/lyh/gnn_project/DiHug')
R=ROOT/'results/typed_structure_192_de_formal_v1_20261028T140346'
S=pathlib.Path(__file__).resolve().parent
O=R/'delivery'; O.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,obj): (O/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
for p in S.iterdir():
 if p.is_file():shutil.copy2(p,O/p.name)
for p in R.iterdir():
 if p.is_file():shutil.copy2(p,O/p.name)
for p in (R/'run').rglob('*'):
 if p.is_file() and p.suffix!='.pt':
  q=O/'run'/p.relative_to(R/'run');q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
shutil.copy2(ROOT/'results/typed_structure_192_de_runtime_v1/typed_structure_192_de_runtime_v1.zip',O/'runtime_source.zip')
a=json.loads((R/'completion_audit.json').read_text()); x=json.loads((R/'run/unified_evaluation.json').read_text())
write('LOCAL_BINARY_ASSETS.json',[dict(p,local_path=str(R/'run'/p['path']),uploaded=False) for p in a['payloads'] if p['path'].endswith('.pt')])
write('audit_command.json',{'command':'CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data2/lyh/miniconda3/envs/sparse_block/bin/python research_branches/typed_structure_192_de_results_v1/audit_completed.py --run results/typed_structure_192_de_formal_v1_20261028T140346/run --source research_branches/typed_structure_192_de_runtime_review/experiments/typed_structure_192_de_runtime_v1 --round1 results/typed_structure_192_oabc_formal_v1_20261028T090624/run --round2 results/typed_structure_192_oabc_seed2_v1_20261028T110507/run --output results/typed_structure_192_de_formal_v1_20261028T140346/completion_audit.json','status':'passed','new_scientific_runs':0})
def flat(obj,prefix=''):
 out={}
 for k,v in obj.items():
  k=prefix+str(k)
  if isinstance(v,dict):out.update(flat(v,k+'.'))
  else:out[k]=json.dumps(v,ensure_ascii=False) if isinstance(v,list) else v
 return out
def csvout(name,rows):
 keys=sorted(set().union(*(r.keys() for r in rows)))
 with (O/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
graphs=[];means=[];genre={}
for cell,c in x['cells'].items():
 means.append(dict(cell=cell,**flat(c['mean'])))
 for seed,g in c['graphs'].items():
  graphs.append(dict(cell=cell,**flat(g)))
  if cell.startswith('IMDB'):
   genre[cell+'_'+seed]={k:v for k,v in g['values']['raw'].items() if k.startswith('Genre_sorted_degree')}
csvout('all_graph_metrics.csv',graphs);csvout('cell_means.csv',means);write('genre_spectra.json',genre)
pairs=[]
for c in x['contrasts']:
 for p in c['pairs']:pairs.append(dict(dataset=c['dataset'],train_seed=c['train_seed'],contrast=c['contrast'],**flat(p)))
 pairs.append(dict(dataset=c['dataset'],train_seed=c['train_seed'],contrast=c['contrast'],sample_seed='mean',**flat({k:v for k,v in c.items() if k not in ['dataset','train_seed','contrast','pairs']})))
csvout('paired_metrics.csv',pairs);csvout('costs.csv',[dict(cell=k,**flat(v)) for k,v in a['cells'].items()])
write('delivery_scope.json',{'packaged_at':datetime.datetime.now().astimezone().isoformat(),'new_graphs':16,'historical_graphs':32,'new_models':8,'updates_per_model':1000,'scientific_failures':0,'private_binary_files_excluded':True,'audit':a['status'],'pair_checks':a['pair_numeric_checks'],'no_new_experiments_authorized':True})
files=sorted(p for p in O.rglob('*') if p.is_file())
(O/'SHA256SUMS').write_text(''.join(sha(p)+'  '+p.relative_to(O).as_posix()+'\n' for p in files))
z=R/'typed_structure_192_de_results_v1.zip'
with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED,6) as f:
 for p in sorted(O.rglob('*')):
  if p.is_file():f.write(p,'typed_structure_192_de_results_v1/'+p.relative_to(O).as_posix())
check=pathlib.Path(tempfile.mkdtemp(prefix='de_results_unpack_'))
with zipfile.ZipFile(z) as f:f.extractall(check)
d=check/'typed_structure_192_de_results_v1'; count=0
for line in (d/'SHA256SUMS').read_text().splitlines():
 h,path=line.split('  ',1);assert sha(d/path)==h;count+=1
assert len(list((d/'run').glob('*/sample_*/graph.npz')))==16
assert not list(d.rglob('*.pt'))
receipt={'status':'passed','files_verified':count,'new_graphs':16,'zip_sha256':sha(z),'bytes':z.stat().st_size,'unpacked':str(d),'full_cpu_audit':'completion_audit.json','no_weight_recertification_from_public_package':True}
(R/'unpack_verification.json').write_text(json.dumps(receipt,indent=2)+'\n');(R/(z.name+'.sha256')).write_text(sha(z)+'  '+z.name+'\n')
print(json.dumps(receipt))
