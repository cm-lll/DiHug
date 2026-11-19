"""Recompute complete PR3B endpoints; never imports the historical project."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import json,hashlib,time,itertools,sys
from pathlib import Path
from collections import Counter
import numpy as np
from scipy import sparse
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def snap(p):
 w=read(p);v=w['payload'];assert digest(v)==w['sha256'];assert digest(v['target_rows'])==v['target_hash'];return v
def parse(g):
 ty=np.array(g['node_types']);n=len(ty);names=sorted(f['name'] for f in g['families']);deg=np.zeros((n,len(names)),dtype=np.int64);edges=set();familyedges={}
 for f in g['families']:
  k=names.index(f['name']);es=set();left=set(f['src_nodes']);right=set(f['dst_nodes'])
  for u,v in f['edges']:
   assert 0<=u<n and 0<=v<n and u!=v
   assert (u in left and v in right) or (v in left and u in right)
   e=tuple(sorted((u,v)));assert e not in edges;edges.add(e);es.add(e);deg[u,k]+=1;deg[v,k]+=1
  familyedges[f['name']]=es
 ii=[u for u,v in edges]+[v for u,v in edges];jj=[v for u,v in edges]+[u for u,v in edges]
 a=sparse.csr_matrix((np.ones(len(ii),dtype=np.int64),(ii,jj)),shape=(n,n))
 components,labels=sparse.csgraph.connected_components(a,directed=False)
 return dict(types=ty,type_names=g['type_names'],degree=deg,adj=a,edges=familyedges,n=n,m=len(edges),components=int(components),LCC=int(np.bincount(labels).max()),family_order=names)
def motifs(g,templates):
 a=g['adj'];ty=g['types'];names=g['type_names'];out=[]
 for template in templates:
  ids=[np.flatnonzero(ty==names.index(x)) for x in template]
  if len(template)==3:
   b=a[ids[0]][:,ids[1]];c=a[ids[1]][:,ids[2]];d=a[ids[2]][:,ids[0]]
   count=int((b@c@d).diagonal().sum());count=count//6 if len(set(template))==1 else count
  else:
   assert len(template)==4 and template[0]==template[2] and template[1]==template[3] and template[0]!=template[1]
   b=a[ids[0]][:,ids[1]];x=sparse.triu(b@b.T,k=1).data;count=int(np.sum(x*(x-1)//2))
  out.append(count)
 return out
def main():
 config=read(ROOT/'configs/endpoint_review_protocol.json');old=Path(config['historical_run']);out=ROOT/'results/r1_endpoints';out.mkdir(exist_ok=False);tic=time.time();cpu=time.process_time()
 manifest=read(old/'manifest.json');state=read(old/'coordinator_state.json');assert not state['workers'] and state['queued_count']==0
 public=read(old/'public_motif_counts.json');assert sha(old/'public_motif_counts.json')==manifest['public_summary_sha']
 sources={str(old/p):h for p,h in manifest['source_files'].items()};assert all(sha(p)==h for p,h in sources.items())
 expected={f'{ds}/s{s}/r{r}/{arm}' for ds in config['datasets'] for s in config['prefix_seeds'] for r in config['repeats'] for arm in config['arms']}
 ledger=read(old/'request_ledger.json');assert set(ledger['requests'])==expected
 entries=[];seedloss={};reference_meta={};arrivals={};loaded=0
 for ds in config['datasets']:
  refpath=Path(manifest['references'][ds]['clean_reference_path']);sources[str(refpath)]=sha(refpath);assert sources[str(refpath)]==manifest['references'][ds]['clean_reference_sha']
  ref=parse(read(refpath));templates=public[ds]['templates'];refcounts=motifs(ref,templates);assert refcounts==public[ds]['counts']
  reference_meta[ds]={k:ref[k] for k in ['n','m','components','LCC','type_names']};reference_meta[ds]['motifs']=refcounts;arrivals[ds]=0;seedloss[ds]={}
  for seed in config['prefix_seeds']:
   prefix=old/ds/f'seed{seed}'/'prefix';report=read(prefix/'final_report.json');sources[str(prefix/'final_report.json')]=sha(prefix/'final_report.json')
   if report['status']=='not_entered':
    for repeat in config['repeats']:
     for arm in config['arms']:
      key=f'{ds}/s{seed}/r{repeat}/{arm}';assert ledger['requests'][key]['status']=='not_entered';entries.append(dict(id=key,status='not_entered'))
    continue
   assert report['status']=='arrived';arrivals[ds]+=1;transition=snap(prefix/'transition.json');start=parse(transition['graph']);seedloss[ds][seed]={a:[] for a in config['arms']}
   for repeat in config['repeats']:
    for arm in config['arms']:
     key=f'{ds}/s{seed}/r{repeat}/{arm}';d=old/ds/f'seed{seed}'/f'r{repeat}'/arm
     for n in ['initial.json','checkpoint_010000.json','final_graph.json','final_report.json']:sources[str(d/n)]=sha(d/n)
     initial=snap(d/'initial.json');final=snap(d/'checkpoint_010000.json');gjson=read(d/'final_graph.json');rep=read(d/'final_report.json')
     assert initial['graph']==transition['graph'] and initial['target_rows']==transition['target_rows'] and initial['completed']==transition['completed']
     assert final['graph']==gjson and final['completed']==rep['completed']==10000
     assert ledger['requests'][key]['status']=='completed' and ledger['requests'][key]['exit_code']==0
     g=parse(gjson);assert g['type_names']==start['type_names'] and np.array_equal(g['types'],start['types'])
     assert g['family_order']==start['family_order'] and np.array_equal(g['degree'],start['degree'])
     counts=motifs(g,templates);loss=sum(abs(a-b)/max(1,b) for a,b in zip(counts,refcounts))/2
     assert counts==rep['final_motifs']==final['motif_cache']['counts'] and abs(loss-rep['main_loss'])<1e-12
     seedloss[ds][seed][arm].append(loss);target=np.array(final['target_rows']);residual=int(np.square(g['degree']-target).sum())
     entries.append(dict(id=key,status='endpoint_checked',motifs=counts,loss=loss,n=g['n'],m=g['m'],components=g['components'],LCC=g['LCC'],relation_degree_preserved=True,target_residual_per_node=residual/g['n'],graph_sha256=sha(d/'final_graph.json')));loaded+=1
   (out/'progress.json').write_text(json.dumps(dict(dataset=ds,seed=seed,checked=loaded)));print(ds,seed,loaded,flush=True)
 rng=np.random.Generator(np.random.PCG64(int.from_bytes(hashlib.sha256(b'PR1/stats/20261007').digest()[:16],'big')));stats={};bounds=[]
 for ds in config['datasets']:
  values=seedloss[ds];ss=sorted(values);stats[ds]=dict(arrived=arrivals[ds],arms={a:float(np.mean([np.mean(values[s][a]) for s in ss])) for a in config['arms']},contrasts={})
  for control in ['R','C']:
   delta=np.array([np.mean(values[s][control])-np.mean(values[s]['M']) for s in ss]);boot=delta[rng.integers(len(ss),size=(20000,len(ss)))].mean(1);lo,hi=np.quantile(boot,[.00625,.99375]);bounds.append(bool(lo>0 and arrivals[ds]>=6))
   stats[ds]['contrasts'][control]=dict(mean=float(delta.mean()),lower=float(lo),upper=float(hi),seed_differences=delta.tolist())
 assert len(entries)==192 and {x['id'] for x in entries}==expected;assert all(sha(p)==h for p,h in sources.items())
 result=dict(endpoint_passed=True,requests=192,checked_outputs=loaded,arrival=arrivals,references=reference_meta,statistics=stats,original_joint_statistic_supported=all(bounds),full_proposal_journal_audit_completed=False,limitation='Endpoint arithmetic and frozen statistical re-evaluation only; proposal chronology and selection policy not fully recertified',records=entries,seed_losses=seedloss,cpu_seconds=time.process_time()-cpu,wall_seconds=time.time()-tic,new_scientific_graphs=0,training=0)
 (out/'report.json').write_text(json.dumps(result,indent=2));(out/'sources.json').write_text(json.dumps(sources,indent=2));print(json.dumps({k:result[k] for k in ['endpoint_passed','checked_outputs','original_joint_statistic_supported','statistics','cpu_seconds','wall_seconds']}),flush=True)
if __name__=='__main__':main()
