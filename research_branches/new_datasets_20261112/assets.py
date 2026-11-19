"""Full-source audit and a single predeclared type-balanced induced sample.

No model call, graph generation, pickle loading or author code execution.
Raw weights/directions remain in source_relations.npz; modelling uses a stated
simple undirected view. Classification labels never enter the model.
"""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import json,time,hashlib,itertools
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from scipy import sparse
from scipy.sparse.csgraph import connected_components
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/external/new_hetero_20261112/prepared'
SEED=2026111201
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def graph_stats(types,edges,schema):
 n=len(types);a=sparse.csr_matrix((np.ones(2*len(edges)),(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(n,n));a.sum_duplicates()
 nc,labels=connected_components(a);sizes=np.bincount(labels);deg=np.diff(a.indptr)
 pairs={tuple(x):i for i,x in enumerate(schema)};f=np.array([pairs[tuple(sorted((int(types[u]),int(types[v]))))] for u,v in edges])
 return dict(nodes=n,edges=len(edges),type_counts=np.bincount(types).tolist(),relation_counts=np.bincount(f,minlength=len(schema)).tolist(),components=int(nc),LCC=int(sizes.max()),isolates=int((deg==0).sum()),max_degree=int(deg.max())),a,labels,f
def extract(types,edges,schema):
 stats,a,components,f=graph_stats(types,edges,schema);rng=np.random.default_rng(SEED);largest=np.argmax(np.bincount(components));available=np.bincount(types[components==largest],minlength=int(types.max())+1)
 assert (available>0).all(),'largest component lacks a type; do not silently change extraction'
 quota=np.zeros(len(available),dtype=int)
 for _ in range(192):
  possible=np.flatnonzero(quota<available);lowest=possible[quota[possible]==quota[possible].min()];quota[int(lowest[0])]+=1
 roots=np.flatnonzero((types==0)&(components==largest));root=int(rng.choice(roots));selected={root};frontier=set(map(int,a.indices[a.indptr[root]:a.indptr[root+1]]));counts=np.bincount([types[root]],minlength=len(available))
 for _ in range(191):
  assert frontier
  candidates=np.array(sorted(frontier));ct=types[candidates];deficit=(quota-counts)/np.maximum(1,quota);best=np.max(deficit[ct]);pool=candidates[deficit[ct]==best];v=int(rng.choice(pool))
  selected.add(v);frontier.remove(v);counts[types[v]]+=1
  frontier.update(int(j) for j in a.indices[a.indptr[v]:a.indptr[v+1]] if int(j) not in selected)
 ids=np.array(sorted(selected));mapping=np.full(len(types),-1,dtype=np.int64);mapping[ids]=np.arange(192);keep=(mapping[edges]>=0).all(1);small=mapping[edges[keep]]
 assert len(ids)==192 and len(np.unique(types[ids]))==len(available)
 smallstats,_,_,sf=graph_stats(types[ids],small,schema);assert smallstats['components']==1
 return ids,small,sf,dict(rule='single seeded connected frontier growth; prefer type with largest remaining proportional quota; choose uniformly among tied frontier nodes; exact induced edges; no outcome selection',seed=SEED,root_original_id=root,quota=quota.tolist(),actual_counts=counts.tolist(),full=stats,small=smallstats)
def prepare(name,names,raw_pairs,source_meta,extra=None):
 out=OUT/name;out.mkdir(parents=True,exist_ok=False)
 counts=source_meta['counts'];types=np.repeat(np.arange(len(names)),counts);offset=np.r_[0,np.cumsum(counts)];schema=np.array(list(itertools.combinations_with_replacement(range(len(names)),2)),dtype=np.int64)
 blocks=[];raw={};audit=[]
 for label,ta,tb,rr,cc,val in raw_pairs:
  rr=np.asarray(rr,dtype=np.int64);cc=np.asarray(cc,dtype=np.int64);val=np.asarray(val,dtype=float)
  assert np.isfinite(val).all() and (val>0).all();assert rr.min()>=0 and rr.max()<counts[ta] and cc.min()>=0 and cc.max()<counts[tb]
  raw[label+'_row']=rr;raw[label+'_col']=cc;raw[label+'_value']=val
  e=np.sort(np.column_stack([rr+offset[ta],cc+offset[tb]]),axis=1);loops=(e[:,0]==e[:,1]);e=np.unique(e[~loops],axis=0);blocks.append(e)
  audit.append(dict(source_matrix=label,types=[names[ta],names[tb]],source_entries=len(rr),undirected_simple_edges=len(e),selfloops_removed=int(loops.sum()),weights_min=float(val.min()),weights_max=float(val.max()),direction_removed=ta==tb))
 edges=np.unique(np.concatenate(blocks),axis=0);assert len(edges)==sum(map(len,blocks)),'same unordered pair has multiple independent relation meanings'
 np.savez_compressed(out/'source_relations.npz',**raw)
 fullstats,_,_,families=graph_stats(types,edges,schema)
 np.savez_compressed(out/'full_graph.npz',types=types,edges=edges,relations=families+1,schema=schema)
 ids,small,sf,extraction=extract(types,edges,schema)
 np.savez_compressed(out/'small_192.npz',types=types[ids],edges=small,relations=sf+1,schema=schema,original_ids=ids)
 if extra is not None:np.savez_compressed(out/'source_id_mapping.npz',**extra)
 # Independent exact induced-subgraph recheck.
 z=np.load(out/'small_192.npz');rev=np.full(len(types),-1);rev[z['original_ids']]=np.arange(192);expected=rev[edges[(rev[edges]>=0).all(1)]]
 assert np.array_equal(z['edges'],expected) and np.array_equal(z['types'],types[z['original_ids']])
 record=dict(dataset=name,type_names=names,source=source_meta,relations=audit,full=fullstats,extraction=extraction,
  task='single-reference undirected typed surrogate generation, NOT original node classification or multi-graph generalization',
  labels_used=False,weights_modelled=False,complete_source_preserved=True,all_declared_nodes_retained_in_full_graph=True,
  schema='one binary undirected relation per unordered type pair; absent-reference families fixed zero under original kernel',
  files={x.name:dict(path=str(x),bytes=x.stat().st_size,sha256=sha(x)) for x in out.glob('*.npz')})
 save(out/'audit.json',record);return record
def main():
 OUT.mkdir(exist_ok=False);start=time.time();results=[]
 src=ROOT/'data/raw/DBLP_four_area/extracted';rows=[line.split() for line in (src/'node.dat').read_text().splitlines() if line.strip()];names=['Paper','Author','Term','Conference'];codes=['P','A','T','C'];ordered=sorted(rows,key=lambda x:(codes.index(x[1]),int(x[0])));mapping={int(x[0]):i for i,x in enumerate(ordered)};types=np.array([codes.index(x[1]) for x in ordered]);counts=np.bincount(types);offset=np.r_[0,np.cumsum(counts)];pairs={}
 for line in (src/'link.dat').read_text().splitlines():
  if not line.strip():continue
  u,v=map(int,line.split());u,v=mapping[u],mapping[v]
  if types[u]>types[v]:u,v=v,u
  a,b=int(types[u]),int(types[v]);pairs.setdefault((a,b),[]).append((u-offset[a],v-offset[b]))
 raw=[]
 for (a,b),es in sorted(pairs.items()):arr=np.array(es);raw.append((f'{names[a]}_{names[b]}',a,b,arr[:,0],arr[:,1],np.ones(len(arr))))
 results.append(prepare('DBLP_HGEN',names,raw,dict(counts=counts.tolist(),origin='HGEN public DBLP_four_area archive, local existing asset; differs from HGB DBLP',repository='https://github.com/lingchen0331/HGEN',license='academic and education use only; no redistribution without separate permission per bundled readme.txt',files={str(src/n):sha(src/n) for n in ['node.dat','link.dat','readme.txt']},archive_sha256=sha(src.parent/'source.zip')),dict(source_ids=np.array([int(x[0]) for x in ordered]),source_labels=np.array([int(x[2]) for x in ordered]))))
 src=ROOT/'data/external/new_hetero_20261112/HAN_ACM.mat';z=loadmat(src);names=['Paper','Author','Term','Venue','Affiliation','Subject','Conference'];counts=[z[k].shape[0] for k in ['P','A','T','V','F','L','C']]
 specs=[('PvsA',0,1),('PvsT',0,2),('PvsV',0,3),('AvsF',1,4),('PvsL',0,5),('PvsC',0,6),('VvsC',3,6),('PvsP',0,0)];raw=[]
 for key,a,b in specs:
  coo=z[key].tocoo();valid=coo.data>0;raw.append((key,a,b,coo.row[valid],coo.col[valid],coo.data[valid]))
 # TvsP is the transpose duplicate; normalized / stopword-pruned matrices are derived views, not additional relation types.
 assert (z['TvsP'].T!=z['PvsT']).nnz==0
 results.append(prepare('ACM_HAN_full7',names,raw,dict(counts=counts,origin='complete HAN ACM.mat direct relation matrices; NOT the commonly reported ACM3025 subset',repository='https://github.com/Jhy1993/HAN',download_manifest=str(src.parent/'download_manifest.json'),sha256=sha(src),license='author repository requests citation; no explicit separate dataset redistribution license located; local research only',excluded='normalized duplicate matrices, stopword-derived duplicate term matrices; metadata strings retained in downloaded MAT; no graph-size filtering')))
 save(OUT/'inventory.json',dict(start_epoch=start,end_epoch=time.time(),datasets=results,extraction_frozen_seed=SEED,GPU=0,model_calls=0))
 print(json.dumps({r['dataset']:{'full':r['full'],'small':r['extraction']['small']} for r in results},indent=2))
if __name__=='__main__':main()
