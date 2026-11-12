#!/usr/bin/env python3
"""Read-only CPU raw-graph evaluation; never trains, samples, matches, or repairs."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import collections,csv,datetime,hashlib,itertools,json,resource,time
from pathlib import Path
import numpy as np
from scipy.special import zeta
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'inputs/reference.npz';RUN=ROOT/'inputs'
OLD_SOURCE=ROOT/'provenance/old_heterogeneous_metrics_71dedb3.py'
NAMES=['Director','Actor','Movie','Genre'];Counter=collections.Counter
REF_SHA='10f1dbbbe55c6a069594b2dcb590d1b8295ca5436320ef4597cd30e6b7f73e42'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,obj):Path(p).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def canon(seq):return min(tuple(seq),tuple(reversed(seq)))
def serial(c):return {','.join(map(str,k)):int(v) for k,v in sorted(c.items())}
def tv(a,b):
 sa,sb=sum(a.values()),sum(b.values())
 if not sa or not sb:return float(bool(sa or sb))
 return .5*sum(abs(a.get(k,0)/sa-b.get(k,0)/sb) for k in a.keys()|b.keys())
def w1(x,y):
 n=int(max(max(x,default=0),max(y,default=0)))+1
 return float(np.abs(np.cumsum(np.bincount(x,minlength=n)/len(x)-np.bincount(y,minlength=n)/len(y))).sum())
def mmd(x,y,alpha=4.):
 sx,cx=np.unique(x,return_counts=True);sy,cy=np.unique(y,return_counts=True);nx,ny=len(x),len(y)
 if min(nx,ny)<2:return dict(raw=None,nonnegative=None)
 xx=np.exp(-alpha*(sx[:,None].astype(float)-sx[None,:])**2);yy=np.exp(-alpha*(sy[:,None].astype(float)-sy[None,:])**2);xy=np.exp(-alpha*(sx[:,None].astype(float)-sy[None,:])**2)
 raw=float((cx@xx@cx-nx)/(nx*(nx-1))+(cy@yy@cy-ny)/(ny*(ny-1))-2*(cx@xy@cy)/(nx*ny));return dict(raw=raw,nonnegative=max(raw,0.))
def powerlaw(degrees,xmin=1):
 x=degrees[degrees>=xmin]
 if len(x)<2:return dict(alpha=None,ks=None,tail_size=len(x),xmin=xmin)
 alpha=1+len(x)/np.log(x/(xmin-.5)).sum();support=np.arange(xmin,x.max()+1,dtype=float)
 ecdf=np.searchsorted(np.sort(x),support,side='right')/len(x);cdf=np.cumsum(support**(-alpha)/zeta(alpha,xmin))
 return dict(alpha=float(alpha),ks=float(np.max(np.abs(ecdf-cdf))),tail_size=len(x),xmin=xmin)
def make_graph(types,edges,schema):
 types=np.asarray(types,np.int64);edges=np.asarray(edges,np.int64).reshape(-1,2);schema=np.asarray(schema,np.int64)
 if len(types)>100000 or len(edges)>1000000:raise ValueError('CPU size guard')
 assert types.ndim==1 and len(set(map(tuple,schema)))==len(schema)
 assert np.all((edges>=0)&(edges<len(types))) and np.all(edges[:,0]!=edges[:,1]);edges=np.sort(edges,axis=1)
 assert len(set(map(tuple,edges)))==len(edges),'Duplicate physical edge; reject not repair'
 lookup={tuple(sorted(pair)):i for i,pair in enumerate(schema)}
 families=np.array([lookup[tuple(sorted((types[u],types[v])))] for u,v in edges],np.int64)
 adj=[{} for _ in types];degree=np.zeros((len(types),len(schema)),np.int64)
 for (u,v),f in zip(edges,families):adj[u][int(v)]=int(f);adj[v][int(u)]=int(f);degree[u,f]+=1;degree[v,f]+=1
 return dict(types=types,edges=edges,schema=schema,families=families,adj=adj,degree=degree,total=degree.sum(1))
def load(path,reference=False):
 with np.load(path,allow_pickle=False) as z:
  edges=z['edges'] if reference else z['edges'].T;g=make_graph(z['types'],edges,z['schema']);f=z['relations']-1 if reference else z['families'];assert np.array_equal(f,g['families'])
 g['source']={'path':str(path),'sha256':sha(path)};return g

def walks(g,length=3,max_ops=20000000):
 types,adj=g['types'],g['adj'];states=[{(int(t),):1} for t in types];out=[];ops=0
 for ell in range(1,length+1):
  nxt=[collections.defaultdict(int) for _ in types];cnt=Counter()
  for u,a in enumerate(adj):
   for v,f in a.items():
    for sig,n in states[u].items():
     ns=sig+(f,int(types[v]));nxt[v][ns]+=n;cnt[canon(ns)]+=n;ops+=1
     if ops>max_ops:raise RuntimeError('walk operation cap; no approximation')
  out.append(cnt);states=nxt
 return out

def simple_paths(g):
 types,adj,edges,fs=g['types'],g['adj'],g['edges'],g['families'];cats=[Counter((int(types[v]),r) for v,r in a.items()) for a in adj];out=[Counter(),Counter(),Counter()]
 for u,a in enumerate(adj):
  items=list(cats[u].items())
  for i,((ta,ra),ca) in enumerate(items):
   for j in range(i,len(items)):
    (tb,rb),cb=items[j];n=ca*cb if i!=j else ca*(ca-1)//2
    if n:out[1][canon((ta,ra,int(types[u]),rb,tb))]+=n
 for (u,v),r in zip(edges,fs):
  u,v,r=int(u),int(v),int(r);out[0][canon((int(types[u]),r,int(types[v])))]+=1;left=cats[u].copy();right=cats[v].copy();left[(int(types[v]),r)]-=1;right[(int(types[u]),r)]-=1
  for (ta,ra),ca in left.items():
   for (tb,rb),cb in right.items():
    if ca and cb:out[2][canon((ta,ra,int(types[u]),r,int(types[v]),rb,tb))]+=ca*cb
  for w in adj[u].keys()&adj[v].keys():out[2][canon((int(types[w]),adj[u][w],int(types[u]),r,int(types[v]),adj[v][w],int(types[w]))) ]-=1
 assert all(v>=0 for c in out for v in c.values());return [+c for c in out]

def stats(g):
 adj,t,d,total=g['adj'],g['types'],g['degree'],g['total'];tri=Counter();pernode=np.zeros(len(t),int)
 for u,a in enumerate(adj):
  for v in a:
   if v<=u:continue
   for w in a.keys()&adj[v].keys():
    if w<=v:continue
    tri[tuple(sorted((int(t[u]),int(t[v]),int(t[w]))))]+=1;pernode[[u,v,w]]+=1
 cluster=np.divide(2*pernode,total*(total-1),out=np.zeros(len(t),float),where=total>1);seen=set();cc=[]
 for start in range(len(t)):
  if start in seen:continue
  todo=[start];seen.add(start);size=0
  while todo:
   u=todo.pop();size+=1
   for v in adj[u]:
    if v not in seen:seen.add(v);todo.append(v)
  cc.append(size)
 assort=None
 if len(g['edges']):
  ed=total[g['edges']].astype(float);mu=ed.mean();var=(ed**2).mean()-mu**2
  if var>0:assort=float((np.prod(ed,axis=1).mean()-mu**2)/var)
 bytype={}
 for typ,name in enumerate(NAMES):
  ix=t==typ;vals=total[ix]
  if not len(vals):continue
  bytype[name]=dict(count=int(ix.sum()),total_degree_histogram=np.bincount(vals).tolist(),joint_relation_degree=serial(Counter(map(tuple,d[ix]))),mean_degree=float(vals.mean()),maximum_degree=int(vals.max()),zeros=int((vals==0).sum()),low_le1=int((vals<=1).sum()),p90=float(np.quantile(vals,.9)),p99=float(np.quantile(vals,.99)),sorted_top10=sorted(map(int,vals),reverse=True)[:10],relation_histograms=[np.bincount(col).tolist() for col in d[ix].T])
 nums=dens=0
 for u in np.flatnonzero(t==2):
  ds=[v for v in adj[u] if t[v]==0];aa=[v for v in adj[u] if t[v]==1];dens+=len(ds)*len(aa);nums+=sum(v in adj[x] for x in ds for v in aa)
 w=walks(g);p=simple_paths(g)
 return dict(nodes=len(t),edges=len(g['edges']),family_edges=np.bincount(g['families'],minlength=len(g['schema'])).tolist(),triangles=int(sum(tri.values())),typed_triangles=serial(tri),average_clustering=float(cluster.mean()),assortativity=assort,lcc=max(cc),components=len(cc),component_sizes=sorted(cc,reverse=True),isolates=int((total==0).sum()),maximum_degree=int(total.max()),total_degree_histogram=np.bincount(total).tolist(),by_type=bytype,walks={str(i+1):dict(counts=serial(c),total=int(sum(c.values()))) for i,c in enumerate(w)},simple_paths={str(i+1):dict(counts=serial(c),total=int(sum(c.values()))) for i,c in enumerate(p)},dma_closure=dict(numerator=nums,denominator=dens,ratio=nums/dens if dens else None),powerlaw_xmin1=powerlaw(total))

def compare(ref,g,rs,gs):
 er=set(map(tuple,ref['edges']));eg=set(map(tuple,g['edges']));overlap=len(er&eg);ty={};fam={}
 for typ,name in enumerate(NAMES):
  ix=ref['types']==typ;dr,dg=ref['degree'][ix],g['degree'][ix];tr,tg=ref['total'][ix],g['total'][ix];threshold=max(1,int(np.ceil(np.quantile(tr,.99))))
  ty[name]=dict(total_degree_W1=w1(tr,tg),relation_degree_W1=[w1(a,b) for a,b in zip(dr.T,dg.T)],joint_relation_degree_TV=tv(Counter(map(tuple,dr)),Counter(map(tuple,dg))),reference_P99_threshold_ge=threshold,reference_above_threshold_count=int((tr>=threshold).sum()),generated_above_threshold_count=int((tg>=threshold).sum()),total_degree_MMD2_rbf_alpha4=mmd(tr,tg))
 for f,(a,b) in enumerate(ref['schema']):
  ix=(ref['types']==a)|(ref['types']==b);fam[str(f)]=dict(endpoint_union_degree_W1=w1(ref['degree'][ix,f],g['degree'][ix,f]),endpoint_union_degree_MMD2_rbf_alpha4=mmd(ref['degree'][ix,f],g['degree'][ix,f]),signed_edge_error=gs['family_edges'][f]-rs['family_edges'][f])
 return dict(total_edge_signed_error=len(eg)-len(er),total_edge_absolute_error=abs(len(eg)-len(er)),total_edge_relative_absolute_error=abs(len(eg)-len(er))/len(er),global_degree_W1=w1(ref['total'],g['total']),degree_MMD2_rbf_alpha4=mmd(ref['total'],g['total']),typed_triangle_TV=tv(rs['typed_triangles'],gs['typed_triangles']),walk_TV={l:tv(rs['walks'][l]['counts'],gs['walks'][l]['counts']) for l in ('1','2','3')},simple_path_TV={l:tv(rs['simple_paths'][l]['counts'],gs['simple_paths'][l]['counts']) for l in ('1','2','3')},walk_total_signed_error={l:gs['walks'][l]['total']-rs['walks'][l]['total'] for l in ('1','2','3')},simple_path_total_signed_error={l:gs['simple_paths'][l]['total']-rs['simple_paths'][l]['total'] for l in ('1','2','3')},edge_overlap_count=overlap,EO_generated_denominator_percent=100*overlap/len(eg) if eg else None,EO_reference_denominator_percent=100*overlap/len(er),EO_jaccard_percent=100*overlap/len(er|eg),exact_labelled_reference_match=er==eg,family_edge_signed_error=(np.asarray(gs['family_edges'])-rs['family_edges']).tolist(),by_type=ty,by_family=fam)

def check_saved(g,s,c,folder):
 m=json.loads((folder/'metrics.json').read_text());oc=json.loads((folder/'comparison.json').read_text());errors=[]
 for k in ['nodes','edges','triangles','lcc','isolates','maximum_degree','average_clustering','assortativity']:
  err=abs(s[k]-m[k]);errors.append(err);assert err<1e-12,(k,err)
 assert s['family_edges']==m['family_edges']
 for i in range(3):
  assert s['simple_paths'][str(i+1)]==dict(counts=m['mp'][i]['counts'],total=m['mp'][i]['total'])
  err=abs(c['simple_path_TV'][str(i+1)]-oc['mp_frequency_TV'][i]);errors.append(err);assert err<1e-12
 for name in NAMES:
  assert s['by_type'][name]['joint_relation_degree']==m['joint_relation_degree'][name]
  err=abs(c['by_type'][name]['joint_relation_degree_TV']-oc['joint_relation_degree_TV'][name]);errors.append(err);assert err<1e-12
  delta=np.abs(np.asarray(c['by_type'][name]['relation_degree_W1'])-oc['typed_relation_degree_W1'][name]);errors.extend(delta);assert max(delta)<1e-12
 return max(errors)
def flatten(row):
 s,c=row['statistics'],row['comparison'];out=dict(arm=row['arm'],seed=row['seed'])
 for k in ['nodes','edges','lcc','triangles','average_clustering','assortativity','isolates','maximum_degree']:out[k]=s[k]
 out.update(total_edge_signed_error=c['total_edge_signed_error'],total_edge_absolute_error=c['total_edge_absolute_error'],total_edge_relative_absolute_error=c['total_edge_relative_absolute_error'],degree_MMD2_alpha4=c['degree_MMD2_rbf_alpha4']['nonnegative'],degree_MMD2_alpha4_raw=c['degree_MMD2_rbf_alpha4']['raw'],typed_triangle_TV=c['typed_triangle_TV'],MP2_walk_TV=c['walk_TV']['2'],MP3_walk_TV=c['walk_TV']['3'],MP2_simple_path_TV=c['simple_path_TV']['2'],MP3_simple_path_TV=c['simple_path_TV']['3'],EO_generated_percent=c['EO_generated_denominator_percent'],EO_reference_percent=c['EO_reference_denominator_percent'],EO_jaccard_percent=c['EO_jaccard_percent'],dma_closure=s['dma_closure']['ratio'],powerlaw_alpha_xmin1=s['powerlaw_xmin1']['alpha'],powerlaw_KS_xmin1=s['powerlaw_xmin1']['ks'])
 for name in NAMES:out[name+'_total_degree_W1']=c['by_type'][name]['total_degree_W1'];out[name+'_joint_role_TV']=c['by_type'][name]['joint_relation_degree_TV']
 for f,(a,b) in enumerate(row['schema']):out[f'edges_{NAMES[a]}_{NAMES[b]}']=s['family_edges'][f]
 return out

def main():
 start=time.monotonic();resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3));resource.setrlimit(resource.RLIMIT_CPU,(180,180));assert sha(SOURCE)==REF_SHA
 ref=load(SOURCE,True);rs=stats(ref);rows=[];graphs={};maxerr=0.
 for arm in ('base','role'):
  for seed in (2026102161,2026102162):
   folder=RUN/arm/'samples_u00006000'/f'seed{seed}';g=load(folder/'graph.npz');su=json.loads((folder/'summary.json').read_text())
   assert su['status']=='completed' and su['next_t']==0 and su['graph_sha256']==g['source']['sha256']
   assert np.array_equal(g['types'],ref['types']) and np.array_equal(g['schema'],ref['schema'])
   s=stats(g);c=compare(ref,g,rs,s);maxerr=max(maxerr,check_saved(g,s,c,folder));graphs[(arm,seed)]=g;rows.append(dict(arm=arm,seed=seed,source=g['source'],schema=g['schema'].tolist(),statistics=s,comparison=c))
 flat=[flatten(r) for r in rows];agg={}
 for arm in ('base','role'):
  fr=[r for r in flat if r['arm']==arm];keys=[k for k in fr[0] if k not in ('arm','seed')];agg[arm]=dict(n_graphs=len(fr),n_train_seeds=1,mean={k:float(np.mean([r[k] for r in fr])) for k in keys},population_std={k:float(np.std([r[k] for r in fr],ddof=0)) for k in keys})
 diversity=[]
 for (ka,a),(kb,b) in itertools.combinations(graphs.items(),2):
  ea=set(map(tuple,a['edges']));eb=set(map(tuple,b['edges']));diff=len(ea)!=len(eb);diversity.append(dict(a=list(ka),b=list(kb),labelled_edge_symmetric_difference=len(ea^eb),labelled_edge_jaccard=len(ea&eb)/len(ea|eb),proven_nonisomorphic=diff,certificate='different physical edge count' if diff else 'not checked'))
 dump(ROOT/'reference.json',dict(source=ref['source'],statistics=rs));dump(ROOT/'per_graph.json',rows);dump(ROOT/'aggregate.json',agg);dump(ROOT/'diversity.json',diversity)
 with (ROOT/'per_graph.csv').open('w') as fp:w=csv.DictWriter(fp,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
 dump(ROOT/'verification.json',dict(verified_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),graphs=len(rows),reference_sha256=REF_SHA,original_metric_source_sha256=sha(OLD_SOURCE),current_metrics_max_abs_error=maxerr,all_original_graph_summary_hashes_verified=True,wall_seconds=time.monotonic()-start,maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,cpu_limit_seconds=180,address_space_limit_gib=2,training=0,new_graphs=0,paper_sample_binding='not established; do not treat matching implementation as submitted-table reproduction'))
 print(json.dumps(agg,indent=2))
if __name__=='__main__':main()
