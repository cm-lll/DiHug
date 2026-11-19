"""Saved-probability CPU audit. No network imports, draws or updates.

Conditional edge-product triangle expectation is exact for the frozen single
query plan, not an expected nonlinear graph loss or joint clean posterior.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import json,csv,time,hashlib,importlib.util,itertools,resource
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'results/hetero_micro_macro_effect_run_20261116T083122'
OLD=ROOT/'results/terminal_structure_real_20261109T233600'
NORMAL=ROOT/'results/probability_shift_check_20261113T104256'
OUT=ROOT/'results/triangle_response_audit_20261116'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(n,x):(OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))
def table(n,rows):
    dump(n+'.json',rows)
    if rows:
        with (OUT/(n+'.csv')).open('w') as f:
            w=csv.DictWriter(f,list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def tri(A):return float(np.sum((A@A)*A)/6)
def matrix(v,edges,n):
    a=np.zeros((n,n));a[edges[0],edges[1]]=v;a[edges[1],edges[0]]=v;return a
def geometry(A):
    d=A.sum(1);W=float(np.sum(d*(d-1))/2);T=tri(A)
    return dict(edges=float(d.sum()/2),triangles=T,wedges=W,closure=3*T/W if W else 0.,
                max_degree=float(d.max()),isolates=int((d==0).sum()))
def main():
    start=time.time();OUT.mkdir(exist_ok=False);bindings={}
    def bind(p):p=Path(p);bindings[str(p)]=sha(p);return p
    def read(p):return json.loads(bind(p).read_text())
    def load(p):return np.load(bind(p))
    frozen=read(SRC/'protocol.json');old=read(OLD/'protocol.json');inv=read(SRC/'inventory.json')
    normal_sources=read(NORMAL/'sources.json')
    dump('protocol.json',dict(scope='64 old actual reverse traces +24 old normal noisy predictions; 48 current final graphs',
        free_models=['IMDB0411','PubMed0421'],normal_models=['IMDB0411','IMDB0412','PubMed0421','PubMed0422'],
        free_sample_seeds=old['sample_seeds'],normal_times=[1,75,250,550,800,1000],
        note='Different time panels; no claim of a matched intervention or current candidate on-policy response',
        bins=[[1,100],[101,400],[401,700],[701,1000]],CN_groups=['0','1-2','3-7','8+'],
        changes='No edits to model/kernel/loss; no new state samples',GPU_hours=0,model_calls=0,max_CPU_seconds=300,
        triangle_decomposition='Expected births by 0/1/2 existing edges; expected destruction of existing 3-edge triangles',
        degree_closure_decomposition='symmetric identity: delta T = avg(C)*delta W/3 + avg(W)*delta C/3; descriptive, not causal'))
    pkg=Path(frozen['parents'][0]['data']).parent.parent
    spec=importlib.util.spec_from_file_location('saved_kernel',bind(pkg/'hsgd/kernel.py'));kernelmodule=importlib.util.module_from_spec(spec);spec.loader.exec_module(kernelmodule)
    context={};terms=[];relrows=[];typerows=[];geom=[];comp=[];refs={};prob_manifest=[]
    for ds in ('IMDB','PubMed'):
        parent=next(r for r in frozen['parents'] if r['dataset']==ds);z=load(parent['data']);ty=z['types'];n=len(ty);schema=z['schema'];e=z['edges'];e=e if e.shape[0]==2 else e.T
        slots=[];fam=[];prior=[];R=matrix(np.ones(e.shape[1]),e,n)
        for ri,(a,b) in enumerate(schema):
            na=np.flatnonzero(ty==a);nb=np.flatnonzero(ty==b)
            qs=list(itertools.combinations(na,2)) if a==b else [tuple(sorted((u,v))) for u in na for v in nb]
            slots.extend(qs);fam.extend([ri]*len(qs));prior.append(sum(R[u,v] for u,v in qs)/len(qs) if qs else 0.)
        slots=np.array(slots).T;fam=np.array(fam);prior=np.array(prior)
        t=np.arange(1001)/1000;alpha=np.cos((t+.008)/1.008*np.pi/2)**2/np.cos(.008/1.008*np.pi/2)**2;alpha[0]=1.;alpha[-1]=0.
        kernel=kernelmodule.MarginalKernel(prior,alpha);Pi=matrix(prior[fam],slots,n)
        tris=np.array(list(itertools.combinations(range(n),3)),dtype=np.int16).T
        compositions=list(itertools.combinations_with_replacement(range(len(np.unique(ty))),3))
        sorted_ty=np.sort(ty[tris],axis=0);tri_type=np.zeros(tris.shape[1],np.int8)
        for k,types in enumerate(compositions):tri_type[np.all(sorted_ty==np.array(types)[:,None],axis=0)]=k
        names=parent['config']['datasets'][ds]['type_names'];refstats=geometry(R);refs[ds]=refstats
        context[ds]=(ty,schema,slots,fam,prior,R,Pi,kernel,tris,compositions,tri_type,names)
    def analyze(ident,x,p0,q,xprev=None,oracle=False):
        if time.time()-start>300:raise TimeoutError('CPU budget')
        ds=ident['dataset'];ty,schema,edges,fam,prior,R,Pi,kernel,tris,compositions,tri_type,names=context[ds];n=len(ty);t=ident['t']
        X=matrix(x,edges,n);P=matrix(p0,edges,n);Q=matrix(q,edges,n);g=geometry(X);CN=X@X
        expected=kernel.reverse(p0,x,fam,t-1,t)
        assert np.max(abs(expected-q))<1e-12
        assert not x[prior[fam]==0].any() and not q[prior[fam]==0].any()
        clean=R[edges[0],edges[1]];oq=kernel.reverse(clean,x,fam,t-1,t)
        OQ=matrix(oq,edges,n);u,v,w=tris
        present=X[u,v]+X[u,w]+X[v,w]
        prob=Q[u,v]*Q[u,w]*Q[v,w]
        assert np.isfinite(prob).all()
        expected_by_k=[float(prob[present==k].sum()) for k in range(4)]
        assert abs(sum(expected_by_k)-tri(Q))<1e-8
        born=sum(expected_by_k[:3]);destroy=g['triangles']-expected_by_k[3]
        D=Q-X;linear=float(np.triu(D*CN,1).sum());quad=float(np.sum(X*(D@D))/2);cubic=tri(D)
        actual_delta=tri(Q)-g['triangles'];assert abs(linear+quad+cubic-actual_delta)<1e-8
        active=(prior[fam]>0);absent=(x==0)&active;existing=(x==1)&active;cn=CN[edges[0],edges[1]]
        births=float(q[absent].sum());deaths=float((1-q[existing]).sum());birthcn=float(np.sum(q[absent]*cn[absent]));deathcn=float(np.sum((1-q[existing])*cn[existing]))
        normalP=kernel.alpha[t]*R+(1-kernel.alpha[t])*Pi
        normalPrev=kernel.alpha[t-1]*R+(1-kernel.alpha[t-1])*Pi
        r=dict(**ident,**g,expected_next_edges=float(q.sum()),edge_births=births,edge_deaths=deaths,
            expected_next_triangles=tri(Q),expected_triangle_change=actual_delta,
            triangle_births=born,triangle_destructions=destroy,
            birth_0_existing=expected_by_k[0],birth_1_existing=expected_by_k[1],birth_2_existing=expected_by_k[2],
            linear_birth_triangle= birthcn,linear_death_triangle=deathcn,quadratic=quad,cubic=cubic,
            birth_weighted_CN=birthcn/births if births else None,death_weighted_CN=deathcn/deaths if deaths else None,
            expected_clean_triangles=tri(P),expected_clean_edges=float(p0.sum()),reference_triangles=tri(R),
            q_forward_triangles=tri(normalP),q_forward_previous_triangles=tri(normalPrev),
            expected_next_triangle_excess=tri(Q)-tri(normalPrev),current_triangle_excess=g['triangles']-tri(normalP),
            oracle_next_triangles=tri(OQ),oracle_triangle_change=tri(OQ)-g['triangles'],
            oracle_note='Reference-identity X0 known only for diagnosis; not an anonymous optimum or available generation oracle',
            actual_triangle_change=tri(matrix(xprev,edges,n))-g['triangles'] if xprev is not None else None)
        r['excess_change']=r['expected_next_triangle_excess']-r['current_triangle_excess']
        terms.append(r)
        # Hold relation composition of births fixed when comparing CN opportunity.
        matched_cn=0.
        for fi,(ta,tb) in enumerate(schema):
            fmask=fam==fi;ab=fmask&absent;ex=fmask&existing;nb=float(q[ab].sum())
            basecn=float(cn[ab].mean()) if ab.any() else 0.;matched_cn+=nb*basecn
            for label,mask in [('absent',ab),('existing',ex)]:
                for group,gg in [('0',cn==0),('1-2',(cn>=1)&(cn<=2)),('3-7',(cn>=3)&(cn<=7)),('8+',cn>=8)]:
                    sel=mask&gg
                    relrows.append(dict(**ident,relation=fi,relation_name=names[ta]+'--'+names[tb],state=label,CN_group=group,
                        support=int(sel.sum()),mean_CN=float(cn[sel].mean()) if sel.any() else None,
                        mean_p0=float(p0[sel].mean()) if sel.any() else None,mean_q=float(q[sel].mean()) if sel.any() else None,
                        expected_births=float(q[sel].sum()) if label=='absent' else 0.,
                        expected_deaths=float((1-q[sel]).sum()) if label=='existing' else 0.,
                        linear_triangle_change=float(np.sum((q[sel]-x[sel])*cn[sel])),
                        reference_edge_count=int(clean[sel].sum()),reference_identity_note='Not a correctness label for anonymous generated nodes'))
        r['relation_matched_absent_CN']=matched_cn/births if births else None
        r['birth_CN_enrichment']=birthcn/matched_cn if matched_cn>0 else None
        for ci,types in enumerate(compositions):
            mask=tri_type==ci;old=float(np.sum(present[mask]==3));new=float(prob[mask].sum())
            typerows.append(dict(**ident,composition='--'.join(names[i] for i in types),triangles=old,
                expected_next_triangles=new,expected_change=new-old,
                births=float(prob[mask&(present<3)].sum()),destructions=old-float(prob[mask&(present==3)].sum())))
    for ds in ('IMDB','PubMed'):
        parent=next(r for r in frozen['parents'] if r['dataset']==ds and r['train_seed']==old['models'][ds]['train_seed'])
        assert sha(bind(parent['parent_readout']))==old['bindings'][parent['parent_readout']]
        for ss in old['sample_seeds']:
            d=OLD/ds/str(ss);identity=read(d/'identity.json')
            assert identity['readout_sha256']==sha(parent['parent_readout']) and identity['train_seed']==parent['train_seed']
            for t in old['times'][str(ss)]:
                fp=d/f'step_{t:04d}.npz';z=load(fp);ty,schema,edges,fam,prior,*_=context[ds]
                assert np.array_equal(z['coverage'],(prior[fam]>0).astype(z['coverage'].dtype))
                np.testing.assert_array_equal(z['xprev'],(z['uniform']<z['q']).astype(z['xprev'].dtype))
                analyze(dict(dataset=ds,seed=parent['train_seed'],kind='saved_reverse',sample_seed=ss,t=t,path=str(fp)),z['xt'],z['p0'],z['q'],z['xprev'])
                prob_manifest.append(str(fp))
    for parent in frozen['parents']:
        ds=parent['dataset'];mid=parent['id'];j=read(NORMAL/(mid+'.json'));z=load(NORMAL/(mid+'_probabilities.npz'))
        assert j['model_unchanged'] and j['RNG_unchanged'] and j['updates']==0
        assert j['states']['baseline']==parent['initial_complete']
        assert sha(bind(parent['initial_complete']))==normal_sources[parent['initial_complete']]==parent['initial_complete_sha256']
        ty,schema,edges,fam,prior,*_=context[ds]
        for k,t in enumerate([1,75,250,550,800,1000]):
            ids=z[f'ids_{k}'];np.testing.assert_array_equal(z[f'query_{k}'],edges[:,ids]);np.testing.assert_array_equal(z[f'family_{k}'],fam[ids])
            x=np.zeros(edges.shape[1],np.uint8);x[ids]=z[f'xt_{k}'];p0=prior[fam].copy();p0[ids]=z[f'baseline_p_{k}'];q=prior[fam].copy();q[ids]=z[f'baseline_q_{k}']
            analyze(dict(dataset=ds,seed=parent['train_seed'],kind='saved_forward_noise',sample_seed=None,t=t,path=str(NORMAL/(mid+'_probabilities.npz'))),x,p0,q)
    # Current endpoints: split wedge opportunity vs closure propensity exactly.
    for item in inv:
        ds=item['dataset'];z=load(item['graph']);e=z['edges'];e=e if e.shape[0]==2 else e.T;ty=context[ds][0];X=matrix(np.ones(e.shape[1]),e,len(ty));g=geometry(X);ref=refs[ds]
        opportunity=(g['closure']+ref['closure'])/2*(g['wedges']-ref['wedges'])/3
        closing=(g['wedges']+ref['wedges'])/2*(g['closure']-ref['closure'])/3
        assert abs(opportunity+closing-(g['triangles']-ref['triangles']))<1e-8
        ident=dict(dataset=ds,seed=item['train_seed'],arm=item['arm'],sample_seed=item['sample_seed'])
        CN=X@X;nt=(CN*X).sum(1)/2;d=X.sum(1)
        geom.append(dict(**ident,**g,reference_triangles=ref['triangles'],reference_wedges=ref['wedges'],reference_closure=ref['closure'],
            opportunity_component=opportunity,closure_component=closing,
            top10_triangle_share=float(np.sort(nt)[-int(np.ceil(.1*len(ty))):].sum()/nt.sum()) if nt.sum() else None))
        u,v,w=context[ds][8];pr=X[u,v]*X[u,w]*X[v,w]
        for ci,types in enumerate(context[ds][9]):
            mask=context[ds][10]==ci;R=context[ds][5]
            comp.append(dict(**ident,composition='--'.join(context[ds][11][i] for i in types),
                triangles=float(pr[mask].sum()),reference_triangles=float((R[u[mask],v[mask]]*R[u[mask],w[mask]]*R[v[mask],w[mask]]).sum())))
    table('response',terms);table('relation_CN',relrows);table('response_composition',typerows);table('terminal_decomposition',geom);table('terminal_composition',comp)
    sums=[]
    for ds,seed,kind in sorted({(r['dataset'],r['seed'],r['kind']) for r in terms}):
        for lo,hi in [(1,100),(101,400),(401,700),(701,1000)]:
            rr=[r for r in terms if (r['dataset'],r['seed'],r['kind'])==(ds,seed,kind) and lo<=r['t']<=hi]
            if not rr:continue
            row=dict(dataset=ds,seed=seed,kind=kind,t_low=lo,t_high=hi,n=len(rr),times=[r['t'] for r in rr])
            for k in ('expected_triangle_change','triangle_births','triangle_destructions','expected_clean_triangles','expected_clean_edges','edge_births','edge_deaths','excess_change','birth_CN_enrichment'):
                vals=[r[k] for r in rr if r[k] is not None];row[k]=float(np.mean(vals)) if vals else None
            sums.append(row)
    table('response_summary',sums)
    assert all(sha(k)==v for k,v in bindings.items())
    dump('asset_manifest.json',dict(bindings=bindings,script_sha256=sha(__file__)))
    dump('receipt.json',dict(status='complete',seconds=time.time()-start,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        saved_reverse=64,saved_forward=24,terminals=48,models_loaded=0,model_calls=0,GPU_hours=0,new_graphs=0,
        posterior_recomputed_agreements=88,old_hard_update_uniform_verified=64,source_unchanged=True,
        limitations=['reverse traces only IMDB0411/PubMed0421','normal panel different times; not matched causal control','new candidate lacks trajectory records']))
    print((OUT/'receipt.json').read_text())
if __name__=='__main__':main()
