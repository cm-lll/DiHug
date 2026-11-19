"""Frozen CPU audit of EXISTING graphs, no model imports or new graph sampling."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import json,csv,time,hashlib,resource
from pathlib import Path
from collections import Counter
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'results/joint_transport_ablation_20261113T011928'
OUT=ROOT/'results/dense_closure_audit_20261113'
hashes={}
def read(p):
    p=Path(p);hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return json.loads(p.read_text())
def load(p):
    p=Path(p);hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return np.load(p)
def csvread(p):
    p=Path(p);hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return list(csv.DictReader(p.open()))
def table(name,rows):
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def dump(name,x):(OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2))
def graph(z):
    ty=z['types'];e=z['edges'];e=e.T if e.shape[0]==2 else e
    es={tuple(sorted(map(int,x))) for x in e}
    assert len(es)==len(e) and all(u!=v for u,v in es)
    a=np.zeros((len(ty),len(ty)),np.int64)
    for u,v in es:a[u,v]=a[v,u]=1
    return ty,a,es
def geometry(a):
    cn=a@a;d=a.sum(1);nt=(cn*a).sum(1)//2
    tris=[]
    for u in range(len(a)):
        for v in np.flatnonzero(a[u,u+1:])+u+1:
            for w in np.flatnonzero(a[u,v+1:]&a[v,v+1:])+v+1:tris.append((u,int(v),int(w)))
    assert sum(nt)==3*len(tris)
    wedges=int((d*(d-1)//2).sum());closure=3*len(tris)/wedges if wedges else 0.
    return d,cn,nt,tris,wedges,closure
def main():
    start=time.time();OUT.mkdir(exist_ok=False)
    p=read(SRC/'protocol.json');eps=sorted((SRC/'evaluation').glob('*.json'));assert len(eps)==64
    reward=csvread(ROOT/'results/robust_heat_transport_20261113/training_rescore.csv')
    evaluations=csvread(ROOT/'results/joint_transport_decomposition_20261113/graphs.csv')
    oldrob=csvread(ROOT/'results/robust_heat_transport_20261113/old_graphs_rescored.csv')
    dump('protocol.json',dict(scope='194 existing graphs: 2 refs, 128 cached train, 64 evaluation',new_graphs=0,model_calls=0,GPU_hours=0,
         frozen_groups='current degree<=1 low; >1 and >=reference type p90 hub; other middle; graph identities not matched to reference',
         paired='shared sample seed parent/final endpoints only; added/removed edges are endpoint differences, not recorded sequential edits',
         decomposition='Tg-Tr=Cr*(Wg-Wr)/3+Wg*(Cg-Cr)/3; descriptive exact identity, not causal or ER null',
         checks='triangle enumeration vs adjacency square; saved counts and source hashes; full typed support',
         max_CPU_seconds=600,evaluation_manifest=[str(x) for x in eps]))
    refs={};cache={};rows=[];nodes=[];compositions=[];relations=[];pairs=[];paired_comp=[]
    for ds in ['IMDB','PubMed']:
        parent=next(v for v in p['parents'].values() if v['dataset']==ds);z=load(parent['data']);ty,a,es=graph(z);g=geometry(a)
        lookup={tuple(sorted(map(int,v))):i for i,v in enumerate(z['schema'])}
        refs[ds]=dict(types=ty,adj=a,edges=es,degree=g[0],wedges=g[4],closure=g[5],triangles=len(g[3]),names=parent['type_names'],schema=z['schema'],lookup=lookup,
                      cut={int(t):float(np.quantile(g[0][ty==t],.9)) for t in np.unique(ty)})
    def process(ds,ident,path,expected=None):
        if time.time()-start>600:raise TimeoutError('CPU bound')
        ref=refs[ds];ty,a,es=graph(load(path));assert np.array_equal(ty,ref['types'])
        assert all(tuple(sorted((int(ty[u]),int(ty[v])))) in ref['lookup'] for u,v in es)
        d,cn,nt,tris,W,C=geometry(a);T=len(tris)
        if expected is not None:assert T==expected
        hub=np.array([d[i]>1 and d[i]>=ref['cut'][int(ty[i])] for i in range(len(d))]);low=d<=1
        H=[x for x in tris if any(hub[list(x)])];H3=[x for x in tris if all(hub[list(x)])]
        r=dict(dataset=ds,**ident,edges=len(es),triangles=T,triangle_error=abs(T-ref['triangles']),wedges=W,transitivity=C,
            degree_opportunity_component=ref['closure']*(W-ref['wedges'])/3,closure_component=W*(C-ref['closure'])/3,
            isolates=int((d==0).sum()),low_count=int(low.sum()),hub_count=int(hub.sum()),max_degree=int(d.max()),
            hub_triangle_participation_fraction=float(nt[hub].sum()/nt.sum()) if nt.sum() else None,
            triangles_touch_hub=len(H),triangles_all_hub=len(H3),hub_edge_count=sum(hub[u] and hub[v] for u,v in es),
            top10_triangle_participation_share=float(np.sort(nt)[-int(np.ceil(.1*len(d))):].sum()/nt.sum()) if nt.sum() else None)
        assert abs(r['degree_opportunity_component']+r['closure_component']-(T-ref['triangles']))<1e-8
        rows.append(r)
        for i in range(len(d)):
            nodes.append(dict(dataset=ds,**ident,node=i,type=ref['names'][int(ty[i])],degree=int(d[i]),triangles=int(nt[i]),group='low' if low[i] else 'hub' if hub[i] else 'middle',local_clustering=float(2*nt[i]/(d[i]*(d[i]-1))) if d[i]>1 else None))
        cc=Counter(tuple(sorted(int(ty[i]) for i in tr)) for tr in tris)
        for labels in sorted(set(cc)|set(tuple(sorted([a,b,c])) for a in range(len(ref['names'])) for b in range(len(ref['names'])) for c in range(len(ref['names'])))):
            compositions.append(dict(dataset=ds,**ident,composition='--'.join(ref['names'][i] for i in labels),triangles=cc[labels]))
        for f,(aa,bb) in enumerate(ref['schema']):
            ee=[(u,v) for u,v in es if tuple(sorted((int(ty[u]),int(ty[v]))))==tuple(sorted((int(aa),int(bb))))]
            relations.append(dict(dataset=ds,**ident,family=f,relation=ref['names'][aa]+'--'+ref['names'][bb],edges=len(ee),triangle_edge_participations=sum(int(cn[u,v]) for u,v in ee)))
        return dict(row=r,adj=a,edges=es,degree=d,cn=cn,triangles=set(tris),hub=hub)
    for ds,ref in refs.items():
        parent=next(v for v in p['parents'].values() if v['dataset']==ds)
        process(ds,dict(panel='reference',seed=0,arm='reference',update=0,sample_seed=0),parent['data'])
    for job in p['jobs']:
        ds=job['dataset'];seed=job['train_seed'];arm=job['architecture'];folder=SRC/job['id']/(ds+'_T_eager')
        for u in range(1,5):
            for plan in job['terminal_plan'][str(u)]:
                ss=plan['seed'];g=process(ds,dict(panel='training',seed=seed,arm=arm,update=u,sample_seed=ss),folder/f'train_round{u}_seed{ss}'/'graph.npz')
                rr=[x for x in reward if x['dataset']==ds and int(x['train_seed'])==seed and x['arm']==arm and int(x['update'])==u and int(x['chain_seed'])==ss];assert len(rr)==1
                g['row'].update(original_score=float(rr[0]['original']),robust_score=float(rr[0]['robust']),robust_degree=float(rr[0]['degree']),robust_heat=float(rr[0]['heat']),reward_advantage=-float(rr[0]['robust_advantage']))
    for ep in eps:
        ev=read(ep);parts=ep.stem.split('_');ds=parts[0];seed=int(parts[1]);arm='_'.join(parts[2:-1]);ss=int(parts[-1]);gp=Path(ev['graph_path']);assert hashlib.sha256(gp.read_bytes()).hexdigest()==ev['graph_sha256']
        gg=process(ds,dict(panel='evaluation',seed=seed,arm=arm,update=4 if arm.endswith('_transport') else 0,sample_seed=ss),gp,ev['raw']['statistics']['triangles'])
        gg['row'].update(original_score=float(ev['unified_transport_score'][0]))
        cache[(ds,seed,arm,ss)]=gg
    for (ds,seed,arm,ss),post in cache.items():
        if not arm.endswith('_transport'):continue
        pre=cache[(ds,seed,arm.replace('_transport',''),ss)];ref=refs[ds];a=pre['adj'];hub=pre['hub'];add=post['edges']-pre['edges'];remove=pre['edges']-post['edges']
        newtri=post['triangles']-pre['triangles'];losttri=pre['triangles']-post['triangles']
        # Match the added-edge relation mixture when comparing available slots.
        weighted_cn=weighted_hub=0.;support=0
        for f,(aa,bb) in enumerate(ref['schema']):
            count=sum(tuple(sorted((int(ref['types'][u]),int(ref['types'][v]))))==tuple(sorted((int(aa),int(bb)))) for u,v in add)
            candidates=[(u,v) for u in range(len(a)) for v in range(u+1,len(a)) if not a[u,v] and tuple(sorted((int(ref['types'][u]),int(ref['types'][v]))))==tuple(sorted((int(aa),int(bb))))]
            if count:
                assert candidates;weighted_cn+=count*np.mean([pre['cn'][u,v] for u,v in candidates]);weighted_hub+=count*np.mean([hub[u] and hub[v] for u,v in candidates]);support+=count
        assert support==len(add)
        ident=dict(dataset=ds,seed=seed,arm=arm,sample_seed=ss)
        pairs.append(dict(**ident,added_edges=len(add),removed_edges=len(remove),net_edges=len(add)-len(remove),new_triangles=len(newtri),lost_triangles=len(losttri),net_triangles=len(newtri)-len(losttri),
            added_edges_parent_CN_mean=float(np.mean([pre['cn'][u,v] for u,v in add])) if add else None,
            relation_matched_nonedge_CN_mean=float(weighted_cn/len(add)) if add else None,
            added_edges_parent_hub_hub_fraction=float(np.mean([hub[u] and hub[v] for u,v in add])) if add else None,
            relation_matched_nonedge_hub_hub_fraction=float(weighted_hub/len(add)) if add else None,
            new_triangles_touch_parent_hub=sum(any(hub[list(tr)]) for tr in newtri),new_triangles_all_parent_hub=sum(all(hub[list(tr)]) for tr in newtri),
            parent_low_to_nonlow=int(((pre['degree']<=1)&(post['degree']>1)).sum()),parent_nonlow_to_low=int(((pre['degree']>1)&(post['degree']<=1)).sum()),
            parent_isolated_to_nonisolated=int(((pre['degree']==0)&(post['degree']>0)).sum()),parent_nonisolated_to_isolated=int(((pre['degree']>0)&(post['degree']==0)).sum()),
            created_triangles_one_new_edge=sum(sum(tuple(sorted(e)) in add for e in [(tr[0],tr[1]),(tr[0],tr[2]),(tr[1],tr[2])])==1 for tr in newtri)))
        tc=Counter(tuple(sorted(ref['names'][int(ref['types'][i])] for i in tr)) for tr in newtri);lc=Counter(tuple(sorted(ref['names'][int(ref['types'][i])] for i in tr)) for tr in losttri)
        for comp in sorted(set(tc)|set(lc)):paired_comp.append(dict(**ident,composition='--'.join(comp),new_triangles=tc[comp],lost_triangles=lc[comp],net_triangles=tc[comp]-lc[comp]))
        assert len(newtri)-len(losttri)==post['row']['triangles']-pre['row']['triangles']
    for name,rr in [('graphs.csv',rows),('nodes.csv',nodes),('triangle_composition.csv',compositions),('relations.csv',relations),('paired_changes.csv',pairs),('paired_triangle_composition.csv',paired_comp)]:table(name,rr)
    assert len(rows)==194 and len(pairs)==32
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    dump('sources.json',hashes);dump('receipt.json',dict(status='complete',graphs=len(rows),pairs=len(pairs),CPU_seconds=time.time()-start,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,GPU_hours=0,model_calls=0,new_graphs=0,source_unchanged=True))
    print((OUT/'receipt.json').read_text())
if __name__=='__main__':main()
