"""CPU-only audit of existing costs, terminal graphs and saved probabilities.

No model imports/calls, no gradient estimation, no new graphs. Terminal-label
votes are only ONE term of the implemented logit descent, not its full gradient.
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import csv,json,hashlib,time,resource
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/reward_direction_audit_20261113'
PANEL=ROOT/'results/joint_transport_ablation_20261113T011928'
PROBS=ROOT/'results/probability_shift_check_20261113T104256'
hashes={}
def bind(p):
    p=Path(p);hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return p
def read(p):return json.loads(bind(p).read_text())
def csvread(p):return list(csv.DictReader(bind(p).open()))
def table(name,rows):
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def corr(a,b):
    return float(np.corrcoef(a,b)[0,1]) if np.std(a)>0 and np.std(b)>0 else None
def edges(z):
    e=z['edges'];e=e.T if e.shape[0]==2 else e
    es={tuple(sorted(map(int,x))) for x in e}
    assert len(es)==len(e) and all(a!=b for a,b in es)
    return es
def main():
    start=time.time();OUT.mkdir(exist_ok=True)
    protocol=read(PANEL/'protocol.json');refs=read(PANEL/'reference_metrics.json')
    costs=csvread(ROOT/'results/robust_heat_transport_20261113/training_rescore.csv')
    graphs=csvread(ROOT/'results/joint_transport_decomposition_20261113/training_graphs.csv')
    key=lambda r:tuple(str(r[k]) for k in ('dataset','train_seed','arm','update','chain_seed'))
    gm={key(r):r for r in graphs};allrows=[];batches=[];families=[];states=[];first=[];node_rows=[]
    # Validate the sign of the implemented a*log Bernoulli likelihood loss.
    zz=.37;aa=-.8;yy0=1.;eps=1e-5
    loss=lambda v:aa*(yy0*(-np.logaddexp(0,-v))+(1-yy0)*(-np.logaddexp(0,v)))
    analytic=aa*(yy0-1/(1+np.exp(-zz)));finite=(loss(zz+eps)-loss(zz-eps))/(2*eps)
    assert abs(analytic-finite)<1e-9
    for job in protocol['jobs']:
        ds=job['dataset'];seed=job['train_seed'];arm=job['architecture'];parent=protocol['parents'][job['parent']]
        refdata=np.load(bind(parent['data']));refedges=edges(refdata)
        names=parent['type_names'];schema=refdata['schema'];refn=len(refedges)
        for u in range(1,5):
            rr=[r for r in costs if r['dataset']==ds and int(r['train_seed'])==seed and r['arm']==arm and int(r['update'])==u]
            assert len(rr)==4
            cc=np.array([float(r['robust']) for r in rr]);adv=(4*cc-cc.sum())/3;reward=-adv
            assert np.allclose(adv,[float(r['robust_advantage']) for r in rr],rtol=0,atol=1e-12)
            assert abs(reward.sum())<1e-12
            vals=[]
            for i,r in enumerate(rr):
                g=gm[key(r)]
                out=dict(dataset=ds,seed=seed,arm=arm,update=u,chain_seed=int(r['chain_seed']),reference_edges=refn,
                         edges=int(g['edges']),isolates=int(g['isolates']),triangles=float(g['triangles']),
                         triangle_error=float(g['triangle_error']),robust_cost=float(r['robust']),degree_cost=float(r['degree']),heat_cost=float(r['heat']),
                         cost_advantage=float(adv[i]),reward_advantage=float(reward[i]))
                allrows.append(out);vals.append(out)
            ne=np.array([v['edges'] for v in vals]);ni=np.array([v['isolates'] for v in vals]);de=np.array([v['degree_cost'] for v in vals]);he=np.array([v['heat_cost'] for v in vals]);da=(4*de-de.sum())/3;ha=(4*he-he.sum())/3
            b=dict(dataset=ds,seed=seed,arm=arm,update=u,reference_edges=refn,mean_edges=float(ne.mean()),
                   target_vote_total=float(reward@ne/4),degree_target_vote=float(-da@ne/4),heat_target_vote=float(-ha@ne/4),
                   reward_edge_corr=corr(reward,ne),reward_isolate_corr=corr(reward,ni),heat_degree_adv_norm_ratio=float(np.linalg.norm(ha)/np.linalg.norm(da)) if np.linalg.norm(da) else None)
            assert abs(b['target_vote_total']-b['degree_target_vote']-b['heat_target_vote'])<1e-9
            batches.append(b)
            if arm!='A' or u!=1:continue
            path=PROBS/(job['id']+'_probabilities.npz');z=np.load(bind(path));q=z['query_0'];fam=z['family_0'];lab=z['label_0'];query=[tuple(sorted(map(int,x))) for x in q.T]
            # R1 covers the same slots but each saved input has its own query order.
            aligned={}
            for j in range(6):
                order={tuple(sorted(map(int,x))):i for i,x in enumerate(z[f'query_{j}'].T)}
                assert len(order)==len(query) and set(order)==set(query)
                ix=np.array([order[e] for e in query])
                assert np.array_equal(z[f'family_{j}'][ix],fam) and np.array_equal(z[f'label_{j}'][ix],lab)
                for v in ['baseline','full','partial']:aligned[f'{v}_p_{j}']=z[f'{v}_p_{j}'][ix]
            yy=[]
            for r in rr:
                path=PANEL/job['id']/(ds+'_T_eager')/f"train_round1_seed{r['chain_seed']}"/'graph.npz';es=edges(np.load(bind(path)));bits=np.array([e in es for e in query],dtype=float)
                assert bits.sum()==len(es);yy.append(bits)
            yy=np.array(yy);vote=reward@yy/4;assert abs(vote.sum()-b['target_vote_total'])<1e-9
            vcommon=np.zeros_like(vote)
            for f in np.unique(fam):
                ix=fam==f;vcommon[ix]=vote[ix].mean();counts=yy[:,ix].sum(1);baseprob=np.mean([aligned[f'baseline_p_{j}'][ix].sum() for j in range(6)])
                a,bb=map(int,schema[int(f)])
                families.append(dict(dataset=ds,seed=seed,family=int(f),relation=f'{names[a]}--{names[bb]}',slots=int(ix.sum()),reference_edges=int(lab[ix].sum()),
                    candidate_edges=counts.tolist(),target_vote=float(vote[ix].sum()),degree_target_vote=float(-da@counts/4),heat_target_vote=float(-ha@counts/4),
                    baseline_expected_edges_six_time_mean=float(baseprob),full_probability_net_change=float(np.mean([(aligned[f'full_p_{j}']-aligned[f'baseline_p_{j}'])[ix].sum() for j in range(6)])),
                    partial_probability_net_change=float(np.mean([(aligned[f'partial_p_{j}']-aligned[f'baseline_p_{j}'])[ix].sum() for j in range(6)]))))
            for j,t in enumerate([1,75,250,550,800,1000]):
                types=z['types_0'];refdeg=np.zeros(len(types));pred={v:np.zeros(len(types)) for v in ['baseline','full','partial']}
                refrel=np.zeros((len(types),len(schema)));pr={v:np.zeros_like(refrel) for v in pred}
                for endpoint in q:
                    np.add.at(refdeg,endpoint,lab)
                    np.add.at(refrel,(endpoint,fam),lab)
                    for v in pred:
                        np.add.at(pred[v],endpoint,aligned[f'{v}_p_{j}'])
                        np.add.at(pr[v],(endpoint,fam),aligned[f'{v}_p_{j}'])
                for variant in ['baseline','full','partial']:
                    p=aligned[f'{variant}_p_{j}'];states.append(dict(dataset=ds,seed=seed,t=t,variant=variant,reference_edges=refn,expected_edges=float(p.sum()),false_mass=float(p[lab==0].sum()),missed_mass=float((1-p[lab==1]).sum())))
                    for typ in np.unique(types):
                        ix=types==typ;d=pred[variant];base=pred['baseline'];step=d-base
                        relevant=np.any(schema==typ,axis=1)
                        joint=cdist(pr[variant][ix][:,relevant],refrel[ix][:,relevant],metric='cityblock')/relevant.sum()
                        ii,jj=linear_sum_assignment(joint);joint_score=float(joint[ii,jj].mean())
                        for group,sel in [('all',ix),('reference_low',ix&(refdeg<=1)),('reference_hub',ix&(refdeg>=np.quantile(refdeg[ix],.9))),('baseline_over',ix&(base>refdeg)),('baseline_under',ix&(base<refdeg))]:
                            node_rows.append(dict(dataset=ds,seed=seed,t=t,variant=variant,type=names[int(typ)],group=group,n=int(sel.sum()),signed_bias=float((d-refdeg)[sel].mean()) if sel.any() else None,
                                degree_MAE=float(abs(d-refdeg)[sel].mean()) if sel.any() else None,degree_change_sum=float(step[sel].sum()),
                                anonymous_expected_degree_W1=float(abs(np.sort(d[ix])-np.sort(refdeg[ix])).mean()) if group=='all' else None,
                                anonymous_joint_expected_relation_degree_L1=joint_score if group=='all' else None))
            first.append(dict(**b,target_vote_common_family_energy=float(vcommon@vcommon/(vote@vote)) if vote@vote else None,
                              terminal_label_only=True,actual_training_probability_term_available=False))
    for name,rows in [('graphs.csv',allrows),('batches.csv',batches),('first_update_relations.csv',families),('fixed_state_edge_masses.csv',states),('first_update_summary.csv',first),('node_allocation.csv',node_rows)]:table(name,rows)
    summary=dict(first_updates=first,all_batches=len(batches),old_graphs=len(allrows),by_dataset_arm={})
    for ds in ['IMDB','PubMed']:
        for arm in ['O','A']:
            rr=[r for r in batches if r['dataset']==ds and r['arm']==arm]
            summary['by_dataset_arm'][ds+'_'+arm]=dict(batches=len(rr),positive_target_vote=sum(r['target_vote_total']>0 for r in rr),negative_target_vote=sum(r['target_vote_total']<0 for r in rr))
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    (OUT/'sources.json').write_text(json.dumps(hashes,indent=2));(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    receipt=dict(status='complete',seconds=time.time()-start,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,GPU_hours=0,model_calls=0,new_graphs=0,source_unchanged=True,scalar_loss_gradient_finite_difference_error=abs(analytic-finite))
    (OUT/'receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(summary,indent=2));print(receipt)
if __name__=='__main__':main()
