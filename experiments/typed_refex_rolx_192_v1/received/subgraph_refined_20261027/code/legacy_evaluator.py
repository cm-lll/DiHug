"""CPU-only exact induced graph audit; no project/model imports."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MPLCONFIGDIR']='/tmp/subgraph_mpl_cache'
os.environ['XDG_CACHE_HOME']='/tmp/subgraph_xdg_cache'
import numpy as np,json,hashlib,csv,ast,itertools,time
from pathlib import Path
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
ROOT=Path(__file__).resolve().parent.parent;OUT=ROOT/'evaluation';OUT.mkdir(exist_ok=True)
SRC=OUT/'audited_paths_excerpt.py'  # bundled audited pure source; no external dependency
if not SRC.exists():SRC=OUT/'audited_paths_excerpt.py'
# Extract the audited, pure functions without executing project imports.
src=SRC.read_text();tree=ast.parse(src);pure='\n\n'.join(ast.get_source_segment(src,n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['canon','paths'])
(OUT/'audited_paths_excerpt.py').write_text('from collections import Counter\n'+pure+'\n')
exec(compile(pure,str(SRC),'exec'),globals())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))
def load(p):
 z=np.load(p);return {k:z[k] for k in z.files}
def stats(x):
 x=np.asarray(x);return dict(count=len(x),mean=float(np.mean(x)) if len(x) else None,median=float(np.median(x)) if len(x) else None,p90=float(np.quantile(x,.9)) if len(x) else None,p95=float(np.quantile(x,.95)) if len(x) else None,max=int(x.max()) if len(x) else None,histogram=np.bincount(x.astype(int)).tolist())
def calc(g,names):
 t,e,r=g['types'],g['edges'],g['relations'].astype(int)-1;n=len(t);d=np.zeros((n,10),int)
 for (u,v),k in zip(e,r):d[u,k]+=1;d[v,k]+=1
 mp,tr,adj=paths(t,e.T,r,max_ops=100000000)
 tri=np.array([sum(len(adj[v].keys() & a.keys()) for v in a)//2 for a in adj]);deg=d.sum(1);cl=np.divide(2*tri,deg*(deg-1),out=np.zeros(n,float),where=deg>1)
 seen=set();comps=[]
 for u in range(n):
  if u in seen:continue
  stack=[u];seen.add(u);num=0
  while stack:
   v=stack.pop();num+=1
   for w in adj[v]:
    if w not in seen:seen.add(w);stack.append(w)
  comps.append(num)
 counts=np.bincount(r,minlength=10)
 result=dict(nodes=n,edges=len(e),type_counts={names[k]:int(sum(t==k)) for k in range(4)},family_edges=counts.tolist(),family_proportions=(counts/max(1,len(e))).tolist(),degree=stats(deg),type_relation_degree={names[k]:[stats(d[t==k,j]) for j in range(10)] for k in range(4)},triangles=int(sum(tr.values())),typed_triangles={','.join(map(str,k)):int(tr[k]) for k in itertools.combinations_with_replacement(range(4),3)},average_clustering=float(cl.mean()),component_sizes=sorted(comps,reverse=True),lcc=max(comps),isolates=int(sum(deg==0)),mp=[dict(total=int(sum(v.values())),counts={','.join(map(str,k)):int(c) for k,c in sorted(v.items())},proportions={','.join(map(str,k)):float(c/max(1,sum(v.values()))) for k,c in sorted(v.items())}) for v in mp])
 return result,dict(degree=d,total_degree=deg,triangle=tri,clustering=cl)
def subset_scope(t,arr,ids,names):
 return dict(nodes=len(ids),type_counts={names[k]:int(sum(t[ids]==k)) for k in range(4)},degree=stats(arr['total_degree'][ids]),type_relation_degree={names[k]:[stats(arr['degree'][ids][t[ids]==k,j]) for j in range(10)] for k in range(4)},mean_node_triangles=float(arr['triangle'][ids].mean()),sum_node_triangles=int(arr['triangle'][ids].sum()),average_clustering=float(arr['clustering'][ids].mean()),note='Nodes selected for sample measured in original graph; triangle incidence sum is NOT a subgraph triangle count.')
def tv(a,b):
 return .5*sum(abs(a.get(k,0)-b.get(k,0)) for k in a.keys()|b.keys())
def spring(g,seed):
 n=len(g['types']);rng=np.random.default_rng(seed);xy=rng.normal(size=(n,2));a=np.zeros((n,n));e=g['edges'];a[e[:,0],e[:,1]]=a[e[:,1],e[:,0]]=1;k=1/np.sqrt(n)
 for it in range(160):
  delta=xy[:,None]-xy[None,:];dist=np.maximum(np.linalg.norm(delta,axis=2),.005);force=(k*k/dist**2-a*dist/k);disp=(delta*force[:,:,None]).sum(1);norm=np.maximum(np.linalg.norm(disp,axis=1),1e-12);temp=.12*(1-it/160);xy+=disp/norm[:,None]*np.minimum(norm,temp)[:,None];xy-=xy.mean(0)
 return xy

def packed_layout(g,seed):
 n=len(g['types']);adj=[set() for _ in range(n)]
 for u,v in g['edges']:adj[u].add(v);adj[v].add(u)
 seen=set();comps=[]
 for u in range(n):
  if u in seen:continue
  todo=[u];seen.add(u);c=[]
  while todo:
   v=todo.pop();c.append(v)
   for w in adj[v]:
    if w not in seen:seen.add(w);todo.append(w)
  comps.append(sorted(c))
 comps.sort(key=lambda c:(-len(c),c[0]));xy=np.zeros((n,2))
 for j,c in enumerate(comps):
  inv=np.full(n,-1);inv[c]=np.arange(len(c));e=g['edges'];mask=(inv[e]>=0).all(1);sub={'types':g['types'][c],'edges':inv[e[mask]]};z=spring(sub,seed+j);z-=z.mean(0);z/=max(np.abs(z).max(),1e-8)
  if j==0:z*=.95
  else:z*=.16;z+=np.array([-0.8+1.6*(j-1)/max(1,len(comps)-2),-1.35])
  xy[c]=z
 return xy

def main():
 allres={};original={};arrays={};graphs={};rows=[];overlaps=[];grid=[]
 for ds in ['IMDB','PubMed']:
  names=json.loads((ROOT/'inputs/recompute/source_release/configs'/f'{ds}.json').read_text())['data']['type_names'];g=load(ROOT/'inputs/reference'/f'{ds}.npz');m,a=calc(g,names);original[ds]=m;arrays[ds]=a
  np.savez_compressed(OUT/f'{ds}_original_pernode.npz',types=g['types'],original_ids=np.arange(len(g['types'])),**a)
  samples=sorted((ROOT/'samples'/ds).glob('*/metadata.json'))
  if len(samples)!=3:raise RuntimeError(f'{ds}: expected 3 completed metadata files, found {len(samples)}')
  for meta in samples:
   md=json.loads(meta.read_text());p=meta.parent;seed=p.name;s=load(p/'graph.npz');
   # Format negotiated with sampler; original IDs must be explicit, never inferred.
   if 'original_ids' in s:ids=s['original_ids'].astype(int)
   elif (p/'original_node_ids.npy').exists():ids=np.load(p/'original_node_ids.npy')
   elif (p/'selected_original_ids.npy').exists():ids=np.load(p/'selected_original_ids.npy')
   elif 'selected_original_ids' in md:ids=np.asarray(md['selected_original_ids'],int)
   elif 'selected_nodes_original' in md:ids=np.asarray(md['selected_nodes_original'],int)
   else:raise RuntimeError(f'No original mapping: {p}')
   inv=np.full(len(g['types']),-1,int);inv[ids]=np.arange(len(ids));mask=(inv[g['edges']]>=0).all(1);ee=inv[g['edges'][mask]];rr=g['relations'][mask]
   exact=np.array_equal(ee,s['edges']) and np.array_equal(rr,s['relations']);assert np.array_equal(np.bincount(s['types'],minlength=4),np.asarray(md['type_quotas']));assert int(seed)==int(md['seed']);assert exact and np.array_equal(g['types'][ids],s['types']) and np.array_equal(g['schema'],s['schema']);assert len(set(ids))==len(ids)
   sm,sa=calc(s,names);ret=np.divide(sa['total_degree'],a['total_degree'][ids],out=np.zeros(len(ids),float),where=a['total_degree'][ids]>0)
   top={}
   for typ,name in enumerate(names):
    members=np.flatnonzero(g['types']==typ);rank=sorted(members,key=lambda u:(-a['total_degree'][u],int(u)));h=set(rank[:max(1,int(np.ceil(len(rank)*.1)))])
    hit=np.array([i for i,u in enumerate(ids) if u in h]);top[name]=dict(original_top10_count=len(h),selected_count=len(hit),selected_original_ids=ids[hit].tolist(),mean_retained_degree_fraction=float(ret[hit].mean()) if len(hit) else None)
   audit=dict(exact_induced=exact,unique_original_ids=True,no_isolates=sm['isolates']==0,nodes_192=len(ids)==192,graph_sha256=sha(p/'graph.npz'),metadata_sha256=sha(meta),metadata_status=md.get('status'),seed=seed)
   np.savez_compressed(OUT/f'{ds}_{seed}_pernode.npz',original_ids=ids,types=s['types'],original_degree=a['degree'][ids],induced_degree=sa['degree'],original_triangles=a['triangle'][ids],induced_triangles=sa['triangle'],original_clustering=a['clustering'][ids],induced_clustering=sa['clustering'],retained_degree_fraction=ret)
   result=dict(degree_sum_retention=float(sa['total_degree'].sum()/a['total_degree'][ids].sum()),audit=audit,selected_nodes_in_original=subset_scope(g['types'],a,ids,names),final_induced=sm,hub_retention=top,retained_degree_fraction=dict(mean=float(ret.mean()),median=float(np.median(ret)),min=float(ret.min()),max=float(ret.max())),mp_composition_TV_vs_original=[tv(x['proportions'],y['proportions']) for x,y in zip(sm['mp'],m['mp'])]);allres[f'{ds}/{seed}']=result;graphs[f'{ds}/{seed}']=(s,ids)
   rows.append(dict(dataset=ds,seed=seed,nodes=len(ids),edges=sm['edges'],triangles=sm['triangles'],clustering=sm['average_clustering'],lcc=sm['lcc'],components=len(sm['component_sizes']),isolates=sm['isolates'],mean_degree=sm['degree']['mean'],selected_original_mean_degree=result['selected_nodes_in_original']['degree']['mean'],mean_degree_retention=float(ret.mean()),degree_sum_retention=result['degree_sum_retention'],mp2=sm['mp'][1]['total'],mp3=sm['mp'][2]['total'],mp2_TV=result['mp_composition_TV_vs_original'][1],mp3_TV=result['mp_composition_TV_vs_original'][2],**{f'n_{k}':v for k,v in sm['type_counts'].items()}));grid.append((ds,seed,s,names))
  keys=[k for k in graphs if k.startswith(ds+'/')]
  for ka,kb in itertools.combinations(keys,2):
   ga,ia=graphs[ka];gb,ib=graphs[kb];na,nb=set(ia),set(ib);ea=set(map(tuple,ia[ga['edges']]));eb=set(map(tuple,ib[gb['edges']]));overlaps.append(dict(dataset=ds,a=ka,b=kb,node_intersection=len(na&nb),node_union=len(na|nb),node_jaccard=len(na&nb)/len(na|nb),edge_intersection=len(ea&eb),edge_union=len(ea|eb),edge_jaccard=len(ea&eb)/len(ea|eb)))
 dump(OUT/'metrics.json',dict(metric_source=str(SRC),metric_source_sha256=sha(SRC),mp_convention='Exact simple undirected paths, reversal identified; original audited paths() extracted without project imports. Families zero-based in metrics, NPZ one-based.',original=original,samples=allres,overlaps=overlaps))
 for name,rr in [('summary',rows),('overlap',overlaps)]:
  with (OUT/f'{name}.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in rr for k in row)));w.writeheader();w.writerows(rr)
 colors=['#0072B2','#E69F00','#009E73','#CC79A7'];fig,axes=plt.subplots(2,3,figsize=(16,10))
 for ax,(ds,seed,g,names) in zip(axes.flat,grid):
  xy=packed_layout(g,7301);ax.add_collection(LineCollection(xy[g['edges']],colors='#808080',linewidths=.4,alpha=.28));
  for typ,name in enumerate(names):sel=g['types']==typ;ax.scatter(*xy[sel].T,s=19,c=colors[typ],label=f'{name} ({sum(sel)})',edgecolors='white',linewidths=.2)
  ax.autoscale();ax.axis('off');ax.set_title(f'{ds} seed {seed}\nN={len(g["types"])} E={len(g["edges"])}');ax.legend(fontsize=7,loc='upper center',bbox_to_anchor=(.5,-.01),ncol=2);ax.set_aspect('equal')
 fig.suptitle('Induced subgraphs — fixed layout seed 7301; disconnected components packed separately; colors=node types');fig.tight_layout();fig.savefig(OUT/'six_graphs.png',dpi=180);plt.close(fig)
 for ds in original:
  fig,axes=plt.subplots(1,3,figsize=(15,4))
  for ax,(key,(g,ids)) in zip(axes,[(k,v) for k,v in graphs.items() if k.startswith(ds+'/')]):
   per=np.load(OUT/f'{key.replace("/","_")}_pernode.npz')
   for x,label in [(arrays[ds]['total_degree'],'All original nodes'),(per['original_degree'].sum(1),'Selected nodes in original'),(per['induced_degree'].sum(1),'Final induced')]:
    v=np.sort(x);ax.step(v,np.arange(1,len(v)+1)/len(v),where='post',label=label)
   ax.set_xscale('symlog',linthresh=1);ax.set_xlabel('Total degree (symlog)');ax.set_ylabel('Empirical CDF');ax.set_title(key.split('/')[1]);ax.grid(alpha=.2)
  axes[0].legend(fontsize=8);fig.suptitle(f'{ds}: selection bias versus degree truncation');fig.tight_layout();fig.savefig(OUT/f'{ds}_degree_comparison.png',dpi=180);plt.close(fig)
 lines=['# 六张诱导子图：独立描述性核验','', '仅CPU统计及可视化；不训练、不采样生成模型。小图不声称代表原图的无偏样本。','', '## 指标口径','- 原图全节点、入选节点在原图中的指标、最终诱导图分别保存。入选节点原图三角数为节点三角参与次数，不是子图三角总数。','- MP2/MP3使用既有审查源码的精确简单无向路径计数（反向视为同一条）；三角全部20种类型组合含零值。','- 图中布局seed固定7301，与抽图seed独立；颜色为节点类型而非学习角色。','', '## 逐图结果','|数据/seed|节点|边|三角|聚类|LCC|孤立|平均度保留比例|MP2组成TV|MP3组成TV|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for r in rows:lines.append(f'|{r["dataset"]}/{r["seed"]}|{r["nodes"]}|{r["edges"]}|{r["triangles"]}|{r["clustering"]:.4f}|{r["lcc"]}|{r["isolates"]}|{r["mean_degree_retention"]:.3f}|{r["mp2_TV"]:.3f}|{r["mp3_TV"]:.3f}|')
 lines+=['','## 解释限制','- 抽样预先强制稀有关系及高低度锚点，IMDB强制4条AA真实边；因此AA比例及对应三角不应视为按原图比例自然复现。','- 所选节点全部保留相互之间的原始边，但连向未选节点的边必然被截断；原图hub未必仍是子图hub。每节点原/新度向量及保留比例已保存。','- 多源扩展偏向可达及高连接节点；无孤立点是设计约束，不是生成模型改善证据。','- 三个seed之间可能共享节点和边，不是三张独立数据集；重叠见overlap.csv。','- 这些预览用于决定小规模工程/机制测试是否合适，不批准未来训练，也不证明大图或PubMed泛化。','', '## 文件','metrics.json完整三口径、类型与关系统计、路径组成、重叠及哈希；*_pernode.npz逐节点；summary.csv汇总；six_graphs.png与两套degree_comparison.png。']
 (OUT/'REPORT_zh.md').write_text('\n'.join(lines));print(json.dumps(rows,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
