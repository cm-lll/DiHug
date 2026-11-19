"""CPU-only audit of stored forward inputs / parent reverse snapshots.

No neural imports, graph draws, training, or generated trajectory replay.
Exact forward moments assume independent binary legal slots, as implemented by
global_noisy_edges. Standardized deviations are descriptive, not Gaussian tests.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
import csv, hashlib, itertools, json, resource, time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'results/hetero_micro_macro_effect_run_20261116T083122'
OUT=ROOT/'results/state_coverage_audit_20261116_v2'

def write(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False))

def table(name,rows):
    write(name+'.json',rows)
    if rows:
        with (OUT/(name+'.csv')).open('w') as f:
            w=csv.DictWriter(f,list(dict.fromkeys(k for r in rows for k in r)))
            w.writeheader();w.writerows(rows)

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def moments(P):
    P2=P@P; Q=P*P
    triangles=float(np.sum(P2*P)/6)
    # Disjoint triangles have independent indicators; shared-edge covariance.
    shared=np.triu(P*(1-P)*(P2*P2-Q@Q),1).sum()
    variance=triangles-float(np.sum((Q@Q)*Q)/6)+float(shared)
    assert variance>=-1e-8
    d=P.sum(1); dv=(P*(1-P)).sum(1)
    return dict(edges=float(P.sum()/2),edge_variance=float((P*(1-P)).sum()/2),
                triangles=triangles,triangle_variance=max(0.,variance),
                wedges=float(np.sum(d*d+dv-d)/2)),d,d*d+dv

def finite_test():
    # Exhaustive four-node Bernoulli graphs: independently computed moments.
    pairs=list(itertools.combinations(range(4),2));p=np.array([.12,.31,.53,.72,.41,.88])
    P=np.zeros((4,4))
    for (u,v),v0 in zip(pairs,p):P[u,v]=P[v,u]=v0
    vals=[];weights=[]
    for bits in itertools.product((0,1),repeat=6):
        A=np.zeros((4,4))
        for (u,v),b in zip(pairs,bits):A[u,v]=A[v,u]=b
        tri=sum(A[u,v]*A[v,w]*A[u,w] for u,v,w in itertools.combinations(range(4),3))
        vals.append(tri);weights.append(np.prod(np.where(bits,p,1-p)))
    mu=np.dot(vals,weights);var=np.dot(np.square(np.array(vals)-mu),weights)
    m,_,_=moments(P)
    np.testing.assert_allclose([m['triangles'],m['triangle_variance']],[mu,var],atol=1e-12)
    return dict(graphs=64,mean=mu,variance=var,passed=True)

def main():
    start=time.time();OUT.mkdir(exist_ok=False)
    write('protocol.json',dict(scope='stored assets only',source=str(SOURCE),
        bins=[[1,100],[101,400],[401,700],[701,1000]],nearby_training_halfwidth=25,
        comparisons='same exact t analytical forward moments; stored training range within +/-25 is supplementary',
        graphs='all 512 unique recent training inputs, 48 validations, 128 old parent snapshots, all 48 terminals',
        exclusions='new-arm intermediate states unavailable; no inference of their drift onset',
        limits='no tail normality, significance, or causal claim; q support differs from empirical coverage',
        model_calls=0,GPU_hours=0,new_graph_draws=0))
    test=finite_test();bindings={};rows=[];groups=[];cache={};sources=[]
    def bind(p):
        p=Path(p);bindings[str(p)]=sha(p);return p
    p=json.loads(bind(SOURCE/'protocol.json').read_text())
    inventory=json.loads(bind(SOURCE/'inventory.json').read_text())
    pkg=Path(p['parents'][0]['data']).parent.parent
    for name in ['hsgd/kernel.py','hsgd/noising.py','hsgd/slots.py','hsgd/bw_preflight/common.py']:
        bind(pkg/name)
    context={}
    for parent in p['parents']:
        ds=parent['dataset']
        if ds not in context:
            z=np.load(bind(parent['data']));ty=z['types'];schema=z['schema'];ee=z['edges']
            ee=ee if ee.shape[0]==2 else ee.T
            n=len(ty);clean=np.zeros((n,n));clean[ee[0],ee[1]]=1;clean+=clean.T
            slots=[];family=[];prior=[]
            for fi,(a,b) in enumerate(schema):
                na=np.flatnonzero(ty==a);nb=np.flatnonzero(ty==b)
                q=list(itertools.combinations(na,2)) if a==b else [tuple(sorted((u,v))) for u in na for v in nb]
                slots.extend(q);family.extend([fi]*len(q));prior.append(sum(clean[u,v] for u,v in q)/len(q) if q else 0)
            slots=np.array(slots).T;family=np.array(family);prior=np.array(prior)
            Pi=np.zeros((n,n));Pi[slots[0],slots[1]]=prior[family];Pi+=Pi.T
            names=parent['config']['datasets'][ds]['type_names']
            context[ds]=(ty,schema,clean,slots,family,prior,Pi,names)
        a=np.load(bind(SOURCE/parent['id']/'CE/inputs.npz'))
        b=np.load(bind(SOURCE/parent['id']/'calibrated_micro_macro/inputs.npz'))
        for split,count in [('train',128),('validation',12)]:
            for i in range(count):
                stem=f'{split}_{i}__';ee=a[stem+'edges'];t=int(a[stem+'t'])
                for key in ('t','edges','edge_families','query','labels'):
                    np.testing.assert_array_equal(a[stem+key],b[stem+key])
                sources.append((dict(dataset=ds,seed=parent['train_seed'],kind=split,arm='shared',sample_seed=None,t=t,index=i,path=str(SOURCE/parent['id']/'CE/inputs.npz')),ee))
    for item in inventory:
        z=np.load(bind(item['graph']));ee=z['edges'];ee=ee if ee.shape[0]==2 else ee.T
        ident={k:item[k] for k in ('dataset','sample_seed','arm')};ident['seed']=item['train_seed']
        sources.append((dict(**ident,kind='terminal',t=0,index=None,path=item['graph']),ee))
        if item['arm']=='parent':
            d=Path(item['directory']);identity=json.loads(bind(d/'identity.json').read_text())
            parent=next(r for r in p['parents'] if r['id']==item['model'])
            assert identity['model_hash']==parent['parent_model_hash'] and identity['updates']==0
            paths=sorted(d.glob('Xt_*.npz'));assert len(paths)==8
            for fp in paths:
                zz=np.load(bind(fp));t=int(zz['t']);bits=zz['xt'];ctx=context[item['dataset']]
                assert len(bits)==ctx[3].shape[1] and np.isin(bits,[0,1]).all()
                sources.append((dict(**ident,kind='parent_reverse',t=t,index=None,path=str(fp)),ctx[3][:,bits.astype(bool)]))
    write('asset_manifest.json',dict(bindings=bindings,states=len(sources)))
    for ident,ee in sources:
        ds=ident['dataset'];ty,schema,clean,slots,family,prior,Pi,names=context[ds]
        n=len(ty);A=np.zeros((n,n));ee=np.asarray(ee,int)
        assert (ee[0]!=ee[1]).all() and len(set(map(tuple,np.sort(ee,axis=0).T)))==ee.shape[1]
        A[ee[0],ee[1]]=1;A[ee[1],ee[0]]=1
        assert not A[Pi==0].any()
        t=ident['t'];key=(ds,t)
        if key not in cache:
            alpha=1. if t==0 else (0. if t==1000 else np.cos((t/1000+.008)/1.008*np.pi/2)**2/np.cos(.008/1.008*np.pi/2)**2)
            P=alpha*clean+(1-alpha)*Pi;m,ed,ed2=moments(P)
            cache[key]=(m,ed,ed2,(P@P)*P,alpha)
        m,ed,ed2,ct,alpha=cache[key];d=A.sum(1);A2=A@A;nt=(A2*A).sum(1)/2
        tri=float(nt.sum()/3);wed=float(np.sum(d*(d-1))/2)
        r=dict(**ident,retention=alpha,edges=float(d.sum()/2),triangles=tri,wedges=wed,
               transitivity=3*tri/wed if wed else None,degree_second_moment=float(np.mean(d*d)),
               maximum_degree=float(d.max()),isolates=int(np.sum(d==0)),q_edges=m['edges'],q_triangles=m['triangles'],
               q_wedges=m['wedges'],q_degree_second_moment=float(ed2.mean()),q_triangle_sd=float(np.sqrt(m['triangle_variance'])),
               q_edge_sd=float(np.sqrt(m['edge_variance'])))
        for metric,var in [('edges','edge_variance'),('triangles','triangle_variance')]:
            r[metric+'_q_z']=(r[metric]-m[metric])/np.sqrt(m[var]) if m[var]>1e-15 else None
        r['triangle_excess']=tri-m['triangles'];r['wedge_excess']=wed-m['wedges']
        rows.append(r)
        for ti,name in enumerate(names):
            ix=ty==ti;dsub=d[ix];k=max(1,int(np.ceil(.1*len(dsub))))
            groups.append(dict(**ident,type=name,nodes=int(ix.sum()),mean_degree=float(dsub.mean()),
                degree_second_moment=float(np.mean(dsub*dsub)),q_degree_second_moment=float(ed2[ix].mean()),
                low_degree_fraction=float(np.mean(dsub<=1)),top10_mean=float(np.sort(dsub)[-k:].mean()),
                triangle_participations=float(nt[ix].sum()),q_triangle_participations=float(ct[ix].sum()/2)))
    table('states',rows);table('type_states',groups)
    bins=[(1,100),(101,400),(401,700),(701,1000)];summaries=[];comparisons=[]
    for ds in context:
        for seed in sorted({r['seed'] for r in rows if r['dataset']==ds}):
            for kind in ('train','validation','parent_reverse'):
                for lo,hi in bins:
                    rr=[r for r in rows if r['dataset']==ds and r['seed']==seed and r['kind']==kind and lo<=r['t']<=hi]
                    if not rr:continue
                    s=dict(dataset=ds,seed=seed,kind=kind,t_low=lo,t_high=hi,n=len(rr))
                    for metric in ('edges','triangles','wedges','q_triangles','triangle_excess','wedge_excess','edges_q_z','triangles_q_z'):
                        vals=[r[metric] for r in rr if r[metric] is not None]
                        s[metric+'_mean']=float(np.mean(vals)) if vals else None
                        s[metric+'_min']=float(min(vals)) if vals else None
                        s[metric+'_max']=float(max(vals)) if vals else None
                    s['triangle_over_3_forward_sd']=int(sum(r['triangles_q_z']>3 for r in rr if r['triangles_q_z'] is not None))
                    summaries.append(s)
            for r in rows:
                if r['dataset']!=ds or r['seed']!=seed or r['kind']!='parent_reverse':continue
                near=[x for x in rows if x['dataset']==ds and x['seed']==seed and x['kind']=='train' and abs(x['t']-r['t'])<=25]
                comparisons.append(dict(dataset=ds,seed=seed,sample_seed=r['sample_seed'],t=r['t'],training_n=len(near),
                    reverse_triangles=r['triangles'],q_triangle_mean=r['q_triangles'],q_triangle_sd=r['q_triangle_sd'],
                    train_triangle_min=min(x['triangles'] for x in near) if near else None,
                    train_triangle_max=max(x['triangles'] for x in near) if near else None,
                    reverse_wedges=r['wedges'],q_wedges=r['q_wedges'],reverse_edges=r['edges'],q_edges=r['q_edges']))
    table('time_bin_summary',summaries);table('nearby_training_comparison',comparisons)
    # Same trajectory paired comparison, never conflate distinct chains as one.
    paths=[]
    for ds in context:
        for seed in sorted({r['seed'] for r in rows if r['dataset']==ds}):
            ss=sorted({r['sample_seed'] for r in rows if r['dataset']==ds and r['seed']==seed and r['kind']=='parent_reverse'})
            for sample in ss:
                rr=sorted([r for r in rows if r['dataset']==ds and r['seed']==seed and r['sample_seed']==sample and r['kind']=='parent_reverse'],key=lambda r:-r['t'])
                low=min(rr,key=lambda r:r['t']);high=max(rr,key=lambda r:r['t'])
                paths.append(dict(dataset=ds,seed=seed,sample_seed=sample,times=[r['t'] for r in rr],
                    triangle_excess_path=[r['triangle_excess'] for r in rr],
                    wedge_excess_path=[r['wedge_excess'] for r in rr],
                    low_t=low['t'],low_triangles=low['triangles'],low_q_triangles=low['q_triangles'],
                    low_edges=low['edges'],low_q_edges=low['q_edges'],
                    low_triangles_q_z=low['triangles_q_z'],high_triangles_q_z=high['triangles_q_z']))
    table('parent_trajectories',paths)
    assert all(sha(k)==v for k,v in bindings.items())
    receipt=dict(status='complete',seconds=time.time()-start,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        states=len(rows),unique_train=512,validation=48,parent_snapshots=128,terminals=48,
        source_assets_unchanged=True,GPU_hours=0,model_calls=0,new_noise_draws=0,finite_test=test,
        unavailable='32 recent CE/candidate chains have no intermediate Xt or structural history; cannot locate their onset',
        support='all active legal configurations possible for 0<t<=1000; probability may be tiny; fixed-prior-zero families remain excluded',
        standardized_deviation='exact q mean/variance, not assumed Gaussian, not independent tests or causal attribution')
    write('receipt.json',receipt);print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
