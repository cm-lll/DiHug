"""Existing graphs/probabilities only: independent joint-CDF values and cost."""
import os,json,time,hashlib,statistics
from pathlib import Path
import numpy as np
import torch
from joint_degree import JointRelationDegreeCDF
torch.set_num_threads(1);torch.set_num_interop_threads(1)
P=Path(__file__).resolve().parents[2];OUT=Path(os.environ['STRUCTURE_RUN']);PROB=P/'results/input_mask_screen_20261106'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(ds):
    path=P/'research_branches/typed_structure_early_cn_expansion_review/experiments/typed_structure_early_cn_expansion_v1/package/assets'/f'{ds}.npz'
    z=np.load(path);types=z['types'];schema=z['schema'];edges=z['edges'].T
    u,v=np.triu_indices(len(types),1);lookup={tuple(r):f for f,r in enumerate(schema)}
    f=np.array([lookup[tuple(sorted((types[a],types[b])))] for a,b in zip(u,v)])
    hard=np.zeros((len(types),len(types)));hard[edges[0],edges[1]]=hard[edges[1],edges[0]]=1
    prior=[float(hard[u[f==i],v[f==i]].mean()) if (f==i).any() else 0 for i in range(len(schema))]
    obj=JointRelationDegreeCDF(types,schema,prior,edges,data_sha256=sha(path))
    active=np.isin(f,[x['relation'] for x in obj.groups])
    return obj,np.array([u[active],v[active]]),f[active]
def independent_joint(obj,p,q,f):
    result=[]
    for g,target in zip(obj.pair_groups,obj.joint_targets):
        cdfs=[]
        for rel,cap in [(g['left_relation'],g['left_capacity']),(g['right_relation'],g['right_capacity'])]:
            rows=[]
            for node in np.flatnonzero(obj.types==g['node_type']):
                ids=np.flatnonzero((f==rel)&((q==node).any(0)));h=np.ones(1)
                for x in p[ids]:h=np.convolve(h,[1-x,x])
                rows.append(np.cumsum(h)[:cap])
            cdfs.append(np.array(rows))
        H=sum(np.outer(a,b) for a,b in zip(*cdfs))/g['nodes']
        result.append(.5*np.mean((H-target)**2))
    return result
def main():
    begun=time.time();objects={d:load(d) for d in ['IMDB','PubMed']};rows=[];graphs=[]
    rng=torch.get_rng_state().clone()
    for b in json.loads((PROB/'bindings.json').read_text()):
        ds=b['task'].split('_')[0];obj,_,_=objects[ds];file=PROB/f'{b["task"]}_{b["kind"]}_{b["t"]}_FULL.npz';z=np.load(file)
        p=z['probabilities'].astype(float);q=z['query'];f=z['family'];reference=np.array(independent_joint(obj,p,q,f))
        for dtype in [torch.float64,torch.float32]:
            x=torch.tensor(p,dtype=dtype,requires_grad=True);tick=time.perf_counter();loss,detail=obj(x,q,f);grad=torch.autograd.grad(loss,x)[0];elapsed=time.perf_counter()-tick
            err=float(np.max(np.abs(reference-detail[len(obj.groups):].detach().numpy())))
            assert err<(1e-12 if dtype==torch.float64 else 1e-7) and torch.isfinite(grad).all()
            rows.append(dict(dataset=ds,task=b['task'],kind=b['kind'],t=b['t'],dtype=str(dtype),marginal_loss=float(detail[:len(obj.groups)].mean()),joint_loss=float(detail[len(obj.groups):].mean()),value_error=err,seconds=elapsed,source_sha256=sha(file)))
    for ds,(obj,q,f) in objects.items():
        for base,arm in [('cn_multiseed_20261104T042546','A'),('pb_degree_formal_20261106T170600','PB')]:
            for task in json.loads((OUT/'tasks.json').read_text()):
                if task['dataset']!=ds:continue
                for ss in [2026102821,2026102822,2026110231,2026110232]:
                    file=P/'results'/base/'run'/task['task_id']/'ema_0.99'/f'sample_{ss}'/'graph.npz';z=np.load(file)
                    adj=np.zeros((obj.n,obj.n));e=z['edges'];adj[e[0],e[1]]=adj[e[1],e[0]]=1
                    _,part=obj(torch.tensor(adj[q[0],q[1]],dtype=torch.float64),q,f)
                    graphs.append(dict(dataset=ds,seed=task['train_seed'],arm=arm,sample_seed=ss,marginal_CDF_error=float(part[:len(obj.groups)].mean()),joint_CDF_error=float(part[len(obj.groups):].mean()),source_sha256=sha(file)))
        (OUT/f'objective_{ds}.json').write_text(json.dumps(obj.objective_binding(),indent=2)+'\n')
    assert torch.equal(rng,torch.get_rng_state()) and not torch.cuda.is_initialized()
    receipt=dict(passed=True,checks=len(rows),graphs=len(graphs),denoiser_calls=0,CUDA_initialized=False,seconds=time.time()-begun,median_CPU_forward_backward={ds:statistics.median(r['seconds'] for r in rows if r['dataset']==ds) for ds in objects},groups={ds:dict(marginal=len(o.groups),joint=len(o.pair_groups)) for ds,(o,_,_) in objects.items()})
    for name,value in [('CPU_receipt',receipt),('CPU_probability_checks',rows),('existing_graph_CDF',graphs)]:
        (OUT/(name+'.json')).write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
