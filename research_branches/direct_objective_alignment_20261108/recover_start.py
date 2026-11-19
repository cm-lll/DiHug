"""Reconstruct only the previously registered start draws, verifying old scores."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import csv,json,hashlib,sys,time,resource
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)
R=Path(__file__).resolve().parents[2];O=R/'results/direct_objective_alignment_20261108'
B=R/'results/direct_structure_response_20261108T205816'
sys.path.insert(0,str(R/'research_branches/hard_structure_rebar_cpu_20261108'))
from objective import JointLocalScore
read=lambda p:json.loads(p.read_text())
P=read(B/'protocol.json');probes=read(B/'probes.json');tasks=read(R/'results/output_boundary_replay_20261108T081654/tasks.json')
tick=time.time();rows=[];bindings={};max_score_difference=0.
def bind(p):bindings[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
def tv(a,b):
    ha={tuple(k):v/len(a) for k,v in zip(*np.unique(a,axis=0,return_counts=True))}
    hb={tuple(k):v/len(b) for k,v in zip(*np.unique(b,axis=0,return_counts=True))}
    return .5*sum(abs(ha.get(k,0)-hb.get(k,0)) for k in ha.keys()|hb.keys())
for ds in ['IMDB','PubMed']:
    p=B/f'{ds}_fixed_inputs.pt';bind(p)
    fixed=torch.load(p,map_location='cpu')['inputs']
    task=next(x for x in tasks if x['dataset']==ds)
    config=task['row']['config']['datasets'][ds]
    data=R/'research_branches/typed_structure_early_cn_expansion_review/experiments/typed_structure_early_cn_expansion_v1/package'/config['file']
    bind(data);assert bindings[str(data)]==config['sha256']
    with np.load(data) as z:types=z['types'].copy()
    for t,seed,inp in fixed:
        rr=next(x for x in probes if x['condition']==ds+'_CE' and x['t']==t)
        nrel=len(rr['relations']);core=JointLocalScore(types,inp['query'],inp['f'],inp['labels'],nrel)
        target=core.features(core.reference).numpy();ref=target[:,nrel:].sum()/3
        for parent in ['CE','PS']:
            condition=ds+'_'+parent;p=B/f'{condition}_start_t{t}.npz';bind(p)
            prob=np.load(p)['probability'];rng=np.random.default_rng(P['evaluation_mc_seeds'][ds]+t)
            saved=next(x for x in probes if x['condition']==condition and x['t']==t and x['arm']=='start')
            for sample in range(32):
                bits=(rng.random(len(prob))<prob).astype(np.float64)
                with torch.no_grad():feature=core.features(torch.from_numpy(bits)).numpy();score=float(core(torch.from_numpy(bits)))
                diff=abs(score-saved['hard_score_samples'][sample]);max_score_difference=max(max_score_difference,diff)
                assert diff<1e-12
                w1=[];rw1=[];jtv=[]
                for ty,ix in core.nodes.items():
                    ix=ix.numpy();mask=core.masks[ty].numpy()[:nrel];a=feature[ix,:nrel];b=target[ix,:nrel]
                    w1.append(float(abs(np.sort(a.sum(1))-np.sort(b.sum(1))).mean()))
                    rw1.append(float(abs(np.sort(a[:,mask],axis=0)-np.sort(b[:,mask],axis=0)).mean()))
                    jtv.append(tv(a[:,mask],b[:,mask]))
                tri=float(feature[:,nrel:].sum()/3)
                rows.append(dict(condition=condition,arm='start',t=t,sample=sample,old_MMD_score=score,
                    type_equal_total_degree_W1=float(np.mean(w1)),type_equal_relation_degree_W1=float(np.mean(rw1)),
                    type_equal_joint_degree_TV=float(np.mean(jtv)),triangles=tri,triangle_abs_error=abs(tri-ref),
                    DIRECT=float((np.mean(w1)/max(1,2*float(inp['labels'].sum())/len(types))+np.mean(jtv)+abs(tri-ref)/max(1,ref))/3)))
with (O/'start_sample_metrics.csv').open('w') as f:
    w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in bindings.items())
(O/'start_reconstruction.json').write_text(json.dumps(dict(count=len(rows),new_random_draws=0,reconstructed_registered_draws=len(rows),
    max_old_score_difference=max_score_difference,seconds=time.time()-tick,max_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
    assets=bindings,CUDA_initialized=torch.cuda.is_initialized(),model_calls=0),indent=2))
print('reconstructed',len(rows),'seconds',time.time()-tick,'score_difference',max_score_difference)
