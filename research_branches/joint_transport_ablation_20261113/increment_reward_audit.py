"""Offline reward/terminal-label audit; no model, gradient or new rollout."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='1'
import csv
import json
import resource
import signal
import sys
import time
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import connected_components
import heat_increment_cpu as inc

assets=inc.assets
ROOT,SRC=inc.ROOT,inc.SRC
OUT=ROOT/'results/increment_reward_audit_20261113_v2'
MODES=('original','increment','amplitude_control')

def dump(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False))

def table(name,rr):
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,list(dict.fromkeys(k for r in rr for k in r)));w.writeheader();w.writerows(rr)

def readcsv(p):
    return list(csv.DictReader(assets.bind(p).open()))

def advantage(x):
    x=np.asarray(x,dtype=float)
    return (len(x)*x-x.sum())/(len(x)-1)

def cosine(x,y):
    norm=np.linalg.norm(x)*np.linalg.norm(y)
    return float(x@y/norm) if norm>1e-20 else None

def metrics(sc,b,d,ref):
    raw=d.sum(1);rd=sc.rd.sum(1);tt=[];marg=[];joint=[];low=[];hub=[]
    for typ in sc.scales:
        ix=sc.types==typ;dr=d[ix];rr=sc.rd[ix];n=int(ix.sum());active=rr.sum(0)>0
        tt.append(float(np.mean(abs(np.sort(raw[ix])-np.sort(rd[ix])))))
        marg.append(float(abs(np.sort(dr[:,active],axis=0)-np.sort(rr[:,active],axis=0)).mean()))
        cc=cdist(dr[:,active],rr[:,active],metric='cityblock')/int(active.sum())
        ii,jj=linear_sum_assignment(cc);joint.append(float(cc[ii,jj].mean()))
        low.append(abs(float((raw[ix]<=1).mean()-(rd[ix]<=1).mean())))
        top=max(1,int(np.ceil(.1*n)));hub.append(abs(float(np.sort(raw[ix])[-top:].mean()-np.sort(rd[ix])[-top:].mean())))
    a=np.zeros((sc.n,sc.n));q=sc.q[:,b.astype(bool)];a[q[0],q[1]]=1;a+=a.T
    nc,labels=connected_components(a,directed=False);lcc=int(np.bincount(labels).max())
    counts=np.bincount(sc.f,weights=b,minlength=sc.r)
    return dict(edges=int(b.sum()),edge_error=abs(int(b.sum())-int(sc.labels.sum())),
        relation_count_L1=float(abs(counts-sc.target).sum()),degree_W1=float(np.mean(tt)),
        marginal_relation_degree_W1=float(np.mean(marg)),joint_relation_degree_L1=float(np.mean(joint)),
        type_low_fraction_error=float(np.mean(low)),type_top10_degree_mean_error=float(np.mean(hub)),
        isolates=int((raw==0).sum()),low_count=int((raw<=1).sum()),LCC=lcc,components=int(nc),
        LCC_error=abs(lcc-ref['LCC']))

def main():
    start=time.monotonic();OUT.mkdir(exist_ok=False)
    resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('600s CPU bound')));signal.alarm(600)
    # Exactly the candidate frozen last ticket; do not refit any rule or scale.
    last=assets.read(ROOT/'results/heat_increment_cpu_20261113/sources.json')
    for p in (Path(inc.__file__),Path(assets.__file__),Path(__file__).with_name('transport.py'),Path(assets.heatmod.__file__)):
        assert assets.sha(p)==last[str(p)];assets.bind(p)
    assets.bind(Path(__file__));assets.bind(Path(__file__).with_name('worker.py'))
    control=assets.read(ROOT/'results/heat_increment_cpu_20261113/control_frozen.json')['coefficients']
    dump('protocol.json',dict(scope='32 existing training batches x4 terminal graphs; all 8 O/A parents and four historical rounds',
        modes=MODES,candidate='unchanged first heat level plus adjacent increments; original bandwidths and degree',
        amplitude_control=control,calibration_fit_count=0,new_graphs=0,model_calls=0,GPU_hours=0,
        advantages='a_k=(4*C_k-sum(C))/3; reward=-a; all four used, no winner-only update',
        label_vote='sum_k reward_k*y_ke/4: terminal-label component only, excludes state-specific -reward*p and parameter Jacobian/Adam',
        metrics='anonymous type-mean total-degree W1, reference-active relation marginal W1 and joint L1 Hungarian; top ceil(0.1*n_type) degree mean error; low<=1 fraction error; component/LCC; existing triangle/hub counts',
        interpretation='descriptive old-trajectory feedback, not new-policy/on-policy gradient or learned outcome',
        CPU_threads=1,memory_GiB=2,max_seconds=600))
    protocol=assets.read(SRC/'protocol.json');cache={};refmetrics={}
    dense=readcsv(ROOT/'results/dense_closure_audit_20261113/graphs.csv')
    dm={(r['dataset'],int(r['seed']),r['arm'],int(r['update']),int(r['sample_seed'])):r for r in dense if r['panel']=='training'}
    for ds in ('IMDB','PubMed'):
        par=next(v for v in protocol['parents'].values() if v['dataset']==ds);sc=assets.load(ds,par);cache[ds]=sc
        a=np.zeros((sc.n,sc.n));q=sc.q[:,sc.labels.astype(bool)];a[q[0],q[1]]=1;a+=a.T
        _,ll=connected_components(a,directed=False)
        refmetrics[ds]={'LCC':int(np.bincount(ll).max())}
    graphs=[];batches=[];associations=[];relations=[];preferred=[];types=[]
    error_keys=('edge_error','relation_count_L1','degree_W1','marginal_relation_degree_W1','joint_relation_degree_L1',
                'type_low_fraction_error','type_top10_degree_mean_error','isolates','LCC_error','triangle_error')
    for job in protocol['jobs']:
        ds=job['dataset'];seed=job['train_seed'];arm=job['architecture'];sc=cache[ds]
        folder=SRC/job['id']/(ds+'_T_eager')
        for u in range(1,5):
            ident=dict(dataset=ds,train_seed=seed,arm=arm,update=u);group=[];ys=[];scores={m:[] for m in MODES};degs={m:[] for m in MODES};heats={m:[] for m in MODES}
            oldbatch=assets.read(folder/f'round{u}_batch.json')
            plans=job['terminal_plan'][str(u)];assert len(plans)==4
            for plan in plans:
                ss=plan['seed'];gp=assets.bind(folder/f'train_round{u}_seed{ss}'/'graph.npz')
                summary=assets.read(gp.parent/'summary.json');assert summary['graph_sha256']==assets.sha(gp)
                b=assets.bits(sc,np.load(gp));ys.append(b);ms,h,d=inc.matrices(sc,b)
                saved=assets.read(folder/f'round{u}_seed{ss}_score.json')
                stat=metrics(sc,b,d,refmetrics[ds]);old=dm[(ds,seed,arm,u,ss)]
                assert stat['edges']==int(old['edges']) and stat['isolates']==int(old['isolates'])
                assert stat['edge_error']==abs(int(old['edges'])-int(sc.labels.sum()))
                stat.update({k:float(old[k]) for k in ('triangles','triangle_error','hub_count','triangles_all_hub','transitivity')})
                r={**ident,'chain_seed':ss,**stat}
                for m in MODES:
                    v,pp,_=inc.solved(ms,control[ds],m)
                    if m=='original':assert abs(v['score']-saved['cost'])<1e-8
                    scores[m].append(v['score']);degs[m].append(v['degree']);heats[m].append(v['heat'])
                    r.update({m+'_'+k:x for k,x in v.items()})
                    types.extend({**ident,'chain_seed':ss,'mode':m,**p} for p in pp)
                group.append(r)
            assert np.allclose(scores['original'],oldbatch['costs'],atol=1e-8,rtol=0)
            assert np.allclose(advantage(scores['original']),oldbatch['advantages'],atol=1e-8,rtol=0)
            yy=np.array(ys,dtype=float);rewards={m:-advantage(scores[m]) for m in MODES}
            bmeta={**ident,'mean_edges':float(np.mean([r['edges'] for r in group])), 'reference_edges':int(sc.labels.sum())}
            for m in MODES:
                a=-rewards[m];da=advantage(degs[m]);ha=advantage(heats[m]);assert np.allclose(a,da+ha,atol=1e-8,rtol=0)
                assert abs(rewards[m].sum())<1e-10
                for k,r in enumerate(group):r[m+'_cost_advantage']=float(a[k]);r[m+'_reward']=float(-a[k])
                norm=float(np.linalg.norm(rewards[m]));d_norm=float(np.linalg.norm(da));h_norm=float(np.linalg.norm(ha))
                best=int(np.argmin(scores[m]));oldbest=int(np.argmin(scores['original']))
                bmeta.update({m+'_reward_norm':norm,m+'_heat_degree_adv_ratio':h_norm/d_norm if d_norm>1e-15 else None,
                    m+'_reward_cosine_original':cosine(rewards[m],rewards['original']),
                    m+'_preferred_changed':bool(best!=oldbest),m+'_preferred_chain':group[best]['chain_seed'],
                    m+'_sign_flips':int(np.count_nonzero(np.sign(a)!=np.sign(advantage(scores['original']))))})
                preferred.append({**ident,'mode':m,'preferred_chain':group[best]['chain_seed'],
                                  **{k:group[best][k] for k in error_keys}})
                for key in error_keys:
                    val=np.array([r[key] for r in group]);center=val-val.mean()
                    alignment=cosine(rewards[m],center)
                    associations.append({**ident,'mode':m,'metric':key,'support':4,
                        'reward_metric_alignment':alignment,'signed_feedback_metric':float(rewards[m]@center/4),
                        'interpretation':'positive associates encouragement with larger error; descriptive, not actual metric update'})
                vote=rewards[m]@yy/4;common=np.zeros_like(vote)
                for f in range(sc.r):
                    ix=sc.f==f;counts=yy[:,ix].sum(1);common[ix]=vote[ix].mean()
                    relations.append({**ident,'mode':m,'family':f,'slots':int(ix.sum()),'reference_edges':float(sc.target[f]),
                        'candidate_counts':json.dumps(counts.tolist()),'mean_count':float(counts.mean()),
                        'terminal_label_vote':float(vote[ix].sum()),'degree_label_vote':float(-da@counts/4),'heat_label_vote':float(-ha@counts/4)})
                assert abs(vote.sum()-rewards[m]@yy.sum(1)/4)<1e-8
                bmeta[m+'_terminal_label_vote']=float(vote.sum())
                bmeta[m+'_family_common_vote_energy']=float(common@common/(vote@vote)) if vote@vote>1e-20 else None
            graphs.extend(group);batches.append(bmeta)
    assert len(graphs)==128 and len(batches)==32
    for name,rr in [('graphs.csv',graphs),('batches.csv',batches),('metric_feedback.csv',associations),('relations.csv',relations),('preferred_descriptive.csv',preferred),('types.csv',types)]:table(name,rr)
    assert 'torch' not in sys.modules
    assert all(assets.sha(p)==h for p,h in assets.hashes.items())
    dump('sources.json',assets.hashes)
    dump('receipt.json',dict(status='complete',graphs=128,batches=32,model_calls=0,GPU_hours=0,new_graphs=0,source_unchanged=True,
        original_costs_and_advantages_reproduced=True,seconds=time.monotonic()-start,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024))
    print((OUT/'receipt.json').read_text())

if __name__=='__main__':
    try:main()
    except BaseException:
        import traceback
        if OUT.exists():dump('failure.json',dict(traceback=traceback.format_exc()))
        raise
