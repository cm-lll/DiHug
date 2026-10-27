"""Bounded CPU-only same-type node swaps; never edit induced edges."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np,json,hashlib,itertools,time,datetime,csv,shutil
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parent.parent
SOURCE=Path('/workspace/shared/subgraph_preview_20261027')
def dump(p,x): Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
 z=np.load(p);return {k:z[k] for k in z.files}
def canon(x):return min(tuple(x),tuple(reversed(x)))
def tv(a,b):return .5*sum(abs(a.get(k,0)-b.get(k,0)) for k in a.keys()|b.keys())
def prop(d):
 s=sum(d.values());return {k:v/max(s,1) for k,v in d.items()}
TRI=list(itertools.combinations_with_replacement(range(4),3));TRII={t:i for i,t in enumerate(TRI)}
PROTOCOL={
 'purpose':'Distribution-focused descriptive refinement, not unbiased sampling or globally optimal search; no model or training',
 'target_nodes':192,'quotas':{'IMDB':[48,56,78,10],'PubMed':[40,74,66,12]},
 'proposal_budget_per_dataset':2000,'rng':'numpy.Generator(PCG64)', 'random_seeds':{'IMDB':2026102711,'PubMed':2026102712},
 'objective':{'formula':'0.20*(relation_TV + typed_logdegree_CDF + clustering_abs_error + typed_triangle_TV + MP2_TV) + 0.05*missing_positive_triangle_fraction',
 'relation_TV':'Total variation of the 10 undirected relation family edge proportions.',
 'typed_logdegree_CDF':'Mean normalized 1D Wasserstein/CDF-L1 distance across all positive-reference type x relation degree cells. Per cell integrate empirical CDF difference over log(1+degree), divide by fixed log(1+maximum reference degree in that cell). Actual raw graph degrees are unchanged. Exact discrete support integration, not histogram binning. Reference max is a fixed integration scale, not a matching target.',
 'clustering_abs_error':'Absolute difference of mean local clustering (degree<2 gives zero), all nodes equally weighted.',
 'typed_triangle_TV':'TV of normalized counts of all 20 unordered node-type triples.',
 'MP2_TV':'TV of exact typed simple undirected length-2 path composition; reversal identified, original audited convention.',
 'missing_positive_triangle_fraction':'Missing positive-reference triangle types divided by number of positive-reference triangle types. This macro penalty supplements count-weighted triangle TV.',
 'MP3':'Not optimized, exact independent final diagnostic only.'},
 'baseline_selection':'Score all three original candidates uniformly. Choose lowest objective among hard-valid candidates; ties lexicographic seed. IMDB additionally requires both reference-positive triangle types (including AAM).',
 'hard_gates':['Exactly192 unique original node IDs and exact fixed type quotas','Exact induced edges/relations/schema inherited unchanged from original, no edge editing','Zero isolates','Every original active relation family remains positive','No loss of any triangle type represented in selected baseline; once a missing type is acquired it may not be lost','No loss of any MP2 type represented in selected baseline'],
 'proposal':'At each of exactly2000 iterations choose node type uniformly; remove uniform selected node of that type; choose replacement uniformly from outside-selected same-type frontier with >=1 current selected neighbor with probability0.9, otherwise uniformly from all outside-selected same-type nodes. If frontier empty use full outside set. Reject invalid swaps, accept strictly improving objective by >1e-12; no extra restarts or budget extension.',
 'constraints_note':'192-node induced subgraph necessarily truncates external edges. Per-type quotas further bound available neighbor degrees. Absolute degree tails and raw triangle/path counts cannot be expected to match full graph; no degree rescaling in saved graphs.',
 'original_protocol_sha256':sha(SOURCE/'protocol.json'),
 'source_graph_sha256':{ds:sha(SOURCE/'inputs/reference'/f'{ds}.npz') for ds in ('IMDB','PubMed')}
}

def fast_metrics(g):
 t=g['types'];e=g['edges'];r=g['relations'].astype(int)-1;n=len(t)
 d=np.zeros((n,10),dtype=int)
 np.add.at(d,(e[:,0],r),1);np.add.at(d,(e[:,1],r),1)
 deg=d.sum(1);adj=[set() for _ in t]
 for u,v in e:adj[u].add(int(v));adj[v].add(int(u))
 tc=np.zeros(20,dtype=int);inc=np.zeros(n,dtype=int)
 for u,a in enumerate(adj):
  for v in a:
   if v>u:
    for w in a & adj[v]:
     if w>v:tc[TRII[tuple(sorted((int(t[u]),int(t[v]),int(t[w]))))]]+=1;inc[u]+=1;inc[v]+=1;inc[w]+=1
 cl=np.divide(2*inc,deg*(deg-1),out=np.zeros(n,float),where=deg>1)
 mp=Counter()
 for typ in range(4):
  dt=d[t==typ];cats=[]
  for rel,(a,b) in enumerate(g['schema']):
   if typ in (a,b):cats.append((int(b if typ==a else a),rel))
  for i,(a,ra) in enumerate(cats):
   for j in range(i,len(cats)):
    b,rb=cats[j];c=int(np.sum(dt[:,ra]*(dt[:,ra]-1)//2)) if i==j else int(np.sum(dt[:,ra]*dt[:,rb]))
    if c:mp[canon((a,ra,typ,rb,b))]+=c
 return dict(d=d,degree=deg,counts=np.bincount(r,minlength=10),triangles=tc,clustering=float(cl.mean()),mp2=dict(mp),adj=adj)

class Objective:
 def __init__(self,g,ref):
  self.types=g['types'];self.ref=ref;self.cells=[]
  for typ in range(4):
   for r in range(10):
    vals=ref['d'][self.types==typ,r];mx=int(vals.max())
    if mx>0:self.cells.append((typ,r,np.sort(vals),np.log1p(mx)))
  self.refmp=prop(ref['mp2']);self.reftri=ref['triangles']/ref['triangles'].sum();self.positive=ref['triangles']>0
 def evaluate(self,t,m):
  distances=[]
  for typ,r,v,scale in self.cells:
   x=np.sort(m['d'][t==typ,r]);support=np.union1d(x,v)
   # x cannot exceed reference cell maximum because it is induced.
   left=support[:-1];diff=np.abs(np.searchsorted(x,left,side='right')/len(x)-np.searchsorted(v,left,side='right')/len(v))
   distances.append(float(np.sum(diff*np.diff(np.log1p(support)))/scale))
  terms=dict(relation_TV=float(.5*np.abs(m['counts']/max(1,m['counts'].sum())-self.ref['counts']/self.ref['counts'].sum()).sum()),typed_logdegree_CDF=float(np.mean(distances)),clustering_abs_error=abs(m['clustering']-self.ref['clustering']),typed_triangle_TV=float(.5*np.abs(m['triangles']/max(1,m['triangles'].sum())-self.reftri).sum()),MP2_TV=tv(prop(m['mp2']),self.refmp),missing_positive_triangle_fraction=float(np.mean(m['triangles'][self.positive]==0)))
  return float(.2*sum(v for k,v in terms.items() if k!='missing_positive_triangle_fraction')+.05*terms['missing_positive_triangle_fraction']),terms

def induced(g,ids):
 inv=np.full(len(g['types']),-1,int);inv[ids]=np.arange(len(ids));mask=(inv[g['edges']]>=0).all(1)
 return dict(types=g['types'][ids],edges=inv[g['edges'][mask]],relations=g['relations'][mask],schema=g['schema'].copy(),original_ids=ids.copy())
def gates(g,ids,m,ds,active,tri_required=None,mp_required=None):
 bad=[]
 if len(ids)!=192 or len(set(ids))!=192:bad.append('nodes_or_uniqueness')
 if np.bincount(g['types'],minlength=4).tolist()!=PROTOCOL['quotas'][ds]:bad.append('type_quotas')
 if np.any(m['degree']==0):bad.append('isolate')
 if np.any(m['counts'][active]==0):bad.append('relation_coverage')
 if tri_required is not None and np.any(m['triangles'][tri_required]==0):bad.append('triangle_coverage')
 if mp_required is not None and any(m['mp2'].get(k,0)==0 for k in mp_required):bad.append('MP2_coverage')
 return bad

def run():
 if (ROOT/'protocol_freeze_receipt.json').exists():raise RuntimeError('Already frozen/run: refusing overwrite')
 PROTOCOL['frozen_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();PROTOCOL['code_sha256']=sha(__file__)
 dump(ROOT/'protocol.json',PROTOCOL);dump(ROOT/'protocol_freeze_receipt.json',dict(frozen_at_utc=PROTOCOL['frozen_at_utc'],protocol_sha256=sha(ROOT/'protocol.json'),code_sha256=sha(__file__),state='Written before baseline scoring and all optimization'))
 print('PROTOCOL FROZEN',flush=True)
 results={}
 for ds in ('IMDB','PubMed'):
  start=time.monotonic();g=load(SOURCE/'inputs/reference'/f'{ds}.npz');ref=fast_metrics(g);obj=Objective(g,ref);active=ref['counts']>0;initial=[]
  for p in sorted((SOURCE/'samples'/ds).glob('*/graph.npz')):
   sg=load(p);ids=np.load(p.parent/'original_node_ids.npy');m=fast_metrics(sg);score,terms=obj.evaluate(sg['types'],m);bad=gates(sg,ids,m,ds,active,ref['triangles']>0 if ds=='IMDB' else None)
   initial.append(dict(seed=p.parent.name,path=str(p),score=score,terms=terms,hard_failures=bad))
  feasible=[x for x in initial if not x['hard_failures']];chosen=min(feasible,key=lambda x:(x['score'],x['seed']));print(ds,'BASELINES',json.dumps(initial),flush=True)
  ids=np.load(Path(chosen['path']).parent/'original_node_ids.npy').copy();sg=induced(g,ids);m=fast_metrics(sg);score,terms=obj.evaluate(sg['types'],m);before=score
  required=m['triangles']>0;mp_required=set(m['mp2']);rng=np.random.default_rng(PROTOCOL['random_seeds'][ds]);fulladj=ref['adj'];accepted=0;rejects=Counter();trace=[];members=[np.flatnonzero(g['types']==k) for k in range(4)];selected=set(map(int,ids))
  # Store baseline before any modification; all source files remain untouched.
  out=ROOT/'samples'/ds;out.mkdir(parents=True,exist_ok=True);np.save(out/'baseline_original_node_ids.npy',ids);np.savez_compressed(out/'baseline_graph.npz',**sg)
  for iteration in range(1,PROTOCOL['proposal_budget_per_dataset']+1):
   typ=int(rng.integers(4));pos=int(rng.choice(np.flatnonzero(sg['types']==typ)));old=int(ids[pos]);outside=[int(u) for u in members[typ] if int(u) not in selected];use_frontier=bool(rng.random()<.9)
   pool=[u for u in outside if fulladj[u]&selected] if use_frontier else outside
   if not pool:pool=outside
   new=int(rng.choice(pool));proposal=ids.copy();proposal[pos]=new;ng=induced(g,proposal);nm=fast_metrics(ng);bad=gates(ng,proposal,nm,ds,active,required,mp_required)
   entry=dict(iteration=iteration,type=typ,removed_original_id=old,added_original_id=new,frontier=use_frontier,hard_failures=bad,accepted=False)
   if bad:
    rejects.update(bad);entry['score']=None
   else:
    ns,nt=obj.evaluate(ng['types'],nm);entry['score']=ns
    if ns<score-1e-12:
     ids,sg,m,score,terms=proposal,ng,nm,ns,nt;selected.remove(old);selected.add(new);required|=m['triangles']>0;accepted+=1;entry['accepted']=True;entry['terms']=terms
    else:rejects['not_improved']+=1
   entry['retained_score']=score;trace.append(entry)
   if iteration%250==0:print(ds,iteration,'accepted',accepted,'score',score,'elapsed',round(time.monotonic()-start,2),flush=True)
  np.savez_compressed(out/'graph.npz',**sg);np.save(out/'original_node_ids.npy',ids)
  with (out/'node_map.csv').open('w') as f:
   w=csv.writer(f);w.writerow(['node_id','original_node_id','type_id']);w.writerows((i,int(u),int(g['types'][u])) for i,u in enumerate(ids))
  meta=dict(dataset=ds,status='refined_valid',seed=PROTOCOL['random_seeds'][ds],baseline_seed=chosen['seed'],type_quotas=PROTOCOL['quotas'][ds],target_nodes=192,original_graph_sha256=sha(SOURCE/'inputs/reference'/f'{ds}.npz'),protocol_sha256=sha(ROOT/'protocol.json'),graph_sha256=sha(out/'graph.npz'),proposals=2000,accepted_swaps=accepted,rejection_counts=dict(rejects),baseline_score=before,final_score=score,final_terms=terms,final_hard_failures=gates(sg,ids,m,ds,active,required,mp_required),positive_triangle_types_required=[list(TRI[i]) for i in np.flatnonzero(required)],mp2_required_count=len(mp_required),original_ids=ids.tolist(),elapsed_seconds=time.monotonic()-start)
  dump(out/'metadata.json',meta);dump(out/'optimization_trace.json',trace);results[ds]=dict(baselines=initial,selected_baseline=chosen,final=meta)
  dump(ROOT/'selection_and_refinement.json',results)
  print(ds,'DONE',json.dumps(meta),flush=True)
 dump(ROOT/'optimization_complete.json',dict(completed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),datasets=list(results),cpu_threads=1,training=False))
if __name__=='__main__':run()
