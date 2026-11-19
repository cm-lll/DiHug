"""CPU-only frozen conditional-graph evaluation after GPU release."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,json,csv,time,hashlib,resource
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)
R=Path(__file__).resolve().parents[2];O=Path(sys.argv[1]);P=json.loads((O/'protocol.json').read_text())
sys.path.insert(0,str(R/'research_branches/hard_structure_rebar_cpu_20261108'));from objective import JointLocalScore
sys.path.insert(0,str(R/'research_branches/direct_structure_cpu_20261108'));from score import DirectScore
tick=time.time();startcpu=time.process_time();probes=json.loads((O/'probes.json').read_text());samples=[];predictions=[];relations=[];nodes=[];cache={}
def write(name,rows):
 if not rows:return
 with (O/name).open('w') as f:
  w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
def tv(a,b):
 h={tuple(k):v/len(a) for k,v in zip(*np.unique(a,axis=0,return_counts=True))};j={tuple(k):v/len(b) for k,v in zip(*np.unique(b,axis=0,return_counts=True))}
 return .5*sum(abs(h.get(k,0)-j.get(k,0)) for k in h.keys()|j.keys())
files={}
for ds in ['IMDB','PubMed']:
 p=O/f'{ds}_inputs.pt';files[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();bundle=torch.load(p,map_location='cpu')
 for panel,items in [('new',bundle['fixed']),('legacy',[dict(index=0,t=500,input=bundle['legacy'])])]:
  for item in items:
   inp=item['input'];r=bundle['relations'];types=bundle['types'];core=JointLocalScore(types,inp['query'],inp['f'],inp['labels'],r)
   target=core.features(core.reference).numpy();deg=target[:,:r].sum(1);tri=target[:,r:].sum()/3
   inventory=core.features(torch.ones_like(core.reference)).numpy()
   cache[ds,panel,item['index']]=(item,core,target,deg,float(tri),max(1,2*float(inp['labels'].sum())/len(types)),inventory)
for n,p in enumerate(probes):
 if time.time()>=P['science_stop_epoch']-10:raise TimeoutError('CPU science deadline; retained completed rows in periodic files')
 path=O/p['path'];files[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
 with np.load(path) as z:prob=z['probability'];ce=z['CE']
 item,core,target,deg,reftri,scale,inventory=cache[p['dataset'],p['panel'],p['index']];inp=item['input'];r=core.r;y=inp['labels'];f=inp['f'];q=inp['query'];types=core.types
 with torch.no_grad():features=core.features(torch.from_numpy(prob)).numpy()
 preddeg=features[:,:r].sum(1);et=features[:,r:].sum()/3
 key={k:p[k] for k in ['condition','dataset','seed','arm','offset','update','panel','index','t']}
 base=dict(**key,CE=float(ce.sum()/(len(types)*(len(types)-1)/2)),positive_CE=float(ce[y==1].mean()) if (y==1).any() else None,
   negative_CE=float(ce[y==0].mean()) if (y==0).any() else None,expected_edges=float(prob.sum()),edge_abs_error=float(abs(prob.sum()-y.sum())),
   expected_degree_MAE=float(abs(preddeg-deg).mean()),expected_triangles=float(et),reference_triangles=reftri,expected_triangle_abs_error=float(abs(et-reftri)))
 predictions.append(base)
 for fam in range(r):
  ix=f==fam;pos=ix&(y==1);neg=ix&(y==0);relevant=(target[:,fam]>0)|(inventory[:,fam]>0)
  relations.append(dict(**key,family=fam,positive_count=int(pos.sum()),negative_count=int(neg.sum()),expected_edges=float(prob[ix].sum()),true_edges=int(pos.sum()),
   missed_positive_mass=float((1-prob[pos]).sum()),false_positive_mass=float(prob[neg].sum()),positive_CE=float(ce[pos].mean()) if pos.any() else None,negative_CE=float(ce[neg].mean()) if neg.any() else None,
   expected_relation_degree_MAE=float(abs(features[relevant,fam]-target[relevant,fam]).mean()) if relevant.any() else None))
 for ty in core.type_values:
  ix=types==ty;low=float(np.quantile(deg[ix],.25));high=float(np.quantile(deg[ix],.9))
  for label,mask in [('all',ix),('low',ix&(deg<=low)),('hub',ix&(deg>low)&(deg>=high)),('middle',ix&(deg>low)&(deg<high))]:
   nodes.append(dict(**key,type=int(ty),group=label,support=int(mask.sum()),degree_MAE=float(abs(preddeg[mask]-deg[mask]).mean()) if mask.any() else None,
    triangle_participation_MAE=float(abs(features[mask,r:]-target[mask,r:]).mean()) if mask.any() else None))
 if p['panel']=='new':
  rng=np.random.default_rng(item['conditional_graph_seed']);bits=(rng.random((32,len(prob)))<prob).astype(float)
  with torch.no_grad():hard=core.features(torch.from_numpy(bits)).numpy()
  for j,feature in enumerate(hard):
   w1=[];rw1=[];joint=[]
   for ty,indices in core.nodes.items():
    indices=indices.numpy();mask=core.masks[ty].numpy()[:r];a=feature[indices,:r];b=target[indices,:r]
    w1.append(float(abs(np.sort(a.sum(1))-np.sort(b.sum(1))).mean()))
    rw1.append(float(abs(np.sort(a[:,mask],axis=0)-np.sort(b[:,mask],axis=0)).mean()))
    joint.append(tv(a[:,mask],b[:,mask]))
   tri=float(feature[:,r:].sum()/3);dw=float(np.mean(w1));jt=float(np.mean(joint));te=abs(tri-reftri)
   samples.append(dict(**key,sample=j,sampled_edges=float(bits[j].sum()),sampled_edge_abs_error=float(abs(bits[j].sum()-y.sum())),DIRECT=(dw/scale+jt+te/max(1,reftri))/3,total_degree_W1=dw,relation_degree_W1=float(np.mean(rw1)),joint_TV=jt,
     triangles=tri,triangle_abs_error=te,degree_component=dw/scale/3,joint_component=jt/3,triangle_component=te/max(1,reftri)/3))
   if j==0:
    original=DirectScore(types,q,f,y,bits[j],r)
    assert abs(original.base-samples[-1]['DIRECT'])<1e-12,'score definition mismatch'
 if (n+1)%40==0:
  write('conditional_samples.csv',samples);write('predictions.csv',predictions);write('relations.csv',relations);write('nodegroups.csv',nodes)
  (O/'cpu_progress.json').write_text(json.dumps(dict(probes=n+1,total=len(probes),samples=len(samples),seconds=time.time()-tick)))
  print('CPU',n+1,'/',len(probes),'seconds',time.time()-tick,flush=True)
write('conditional_samples.csv',samples);write('predictions.csv',predictions);write('relations.csv',relations);write('nodegroups.csv',nodes)
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in files.items())
receipt=dict(status='complete',probes=len(probes),conditional_samples=len(samples),seconds=time.time()-tick,cpu_seconds=time.process_time()-startcpu,max_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
 CUDA_initialized=torch.cuda.is_initialized(),assets_unchanged=True)
if json.loads((O/'receipt.json').read_text())['status']=='complete':assert len(probes)==576 and len(samples)==17920
(O/'cpu_receipt.json').write_text(json.dumps(receipt,indent=2));(O/'probability_binding.json').write_text(json.dumps(files,indent=2));print(receipt)
