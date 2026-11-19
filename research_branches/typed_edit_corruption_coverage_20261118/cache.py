"""Paired degree-preserving corruption coverage; every label is a true one-step inverse."""
import importlib.util
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
_spec=importlib.util.spec_from_file_location('coverage_cpu_adapter',ROOT/'research_branches/typed_edit_alignment_audit_20261118/audit.py')
a=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(a)
b=a.b
import metrics
from dihug.core.gpu_role_guided_edit import GpuEditBatch
from dihug.core.mixed_edit_diffusion import EditProposal,mixed_edit_candidate_matrix
OUT=ROOT/'results/typed_edit_corruption_coverage_20261118'


def put(name,value):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def take(batch,index):return GpuEditBatch(**{k:v[index].clone() for k,v in vars(batch).items()})
def inverse(batch):
    return GpuEditBatch(action=batch.action.clone(),family=batch.family.clone(),
        old_edges=batch.new_edges.clone(),new_edges=batch.old_edges.clone(),old_indices=batch.old_indices.clone(),
        old_valid=batch.new_valid.clone(),new_valid=batch.old_valid.clone())


def key(values,i):
    return (values['action'][i],values['family'][i],
        tuple(sorted(tuple(sorted(e)) for e,yes in zip(values['old_edges'][i],values['old_valid'][i]) if yes)),
        tuple(sorted(tuple(sorted(e)) for e,yes in zip(values['new_edges'][i],values['new_valid'][i]) if yes)))


def proposals(graph,values):
    names=sorted(graph.family_specs);actions=['noop','birth','death','relocation','switch']
    return [EditProposal(actions[act],names[values['family'][i]],
        tuple(tuple(e) for e,yes in zip(values['old_edges'][i],values['old_valid'][i]) if yes),
        tuple(tuple(e) for e,yes in zip(values['new_edges'][i],values['new_valid'][i]) if yes))
        for i,act in enumerate(values['action'])]


@torch.no_grad()
def trajectory(kind,tid,steps=128):
    ref=b.graph('pubmed');target=np.column_stack([ref.family_degrees(f) for f in sorted(ref.family_specs)])
    state=a.CpuAuditState(ref,target,'cpu',use_triton=False)
    original_degree=state.family_degree.clone()
    close=kind=='B' and tid%2==0
    xs=[];ys=[];records=[];path=[];max_error=0.;start=time.monotonic()
    for step in range(1,steps+1):
        seed=2026111800000+tid*10000+step
        candidates=state.random_candidates(16,torch.Generator().manual_seed(seed),role_phase=False,
            targeted_fraction=0.,closure_fraction=float(close),candidate_mode='legacy_pool')
        choice=int(torch.randint(1,16,(1,),generator=torch.Generator().manual_seed(seed+5000000)))
        chosen=take(candidates,slice(choice,choice+1));assert int(chosen.action[0])==4
        forward_delta=int(state._closure_components(chosen)[0].sum())
        if step%4==0:
            before_edges=state.edge_tables.clone();before_adj=state.adjacency.clone()
        state.commit(chosen,torch.tensor(0))
        assert torch.equal(state.family_degree,original_degree)
        path.append(dict(step=step,forward_delta=forward_delta,edit=a.serialize(chosen)))
        if step%4:continue
        inv=inverse(chosen);inverse_delta=int(state._closure_components(inv)[0].sum())
        assert inverse_delta==-forward_delta
        after_edges=state.edge_tables.clone();after_adj=state.adjacency.clone()
        state.commit(inv,torch.tensor(0))
        assert torch.equal(state.edge_tables,before_edges) and torch.equal(state.adjacency,before_adj)
        state.commit(chosen,torch.tensor(0))
        assert torch.equal(state.edge_tables,after_edges) and torch.equal(state.adjacency,after_adj)
        assert torch.equal(state.family_degree,original_degree)
        candidates=state.random_candidates(16,torch.Generator().manual_seed(seed+9000000),role_phase=False,
            targeted_fraction=0.,closure_fraction=0.,candidate_mode='legacy_pool')
        values=a.serialize(candidates);wanted=key(a.serialize(inv),0)
        hits=[i for i in range(16) if key(values,i)==wanted]
        assert len(hits)<=1
        label=hits[0] if hits else 15
        if not hits:
            for name,value in vars(candidates).items():value[15]=getattr(inv,name)[0]
        values=a.serialize(candidates)
        assert sum(key(values,i)==wanted for i in range(16))==1
        x=state.compact_features(candidates,step/128,45).numpy()
        graph=state.to_graph(ref)
        cpu=mixed_edit_candidate_matrix(graph,proposals(graph,values),step/128,'typed_closure_compact_v2')
        error=float(np.max(abs(x-cpu)));max_error=max(max_error,error);assert error<=1e-5,error
        assert int(round(x[label,43]*np.sqrt(ref.num_nodes)))==inverse_delta
        assert np.isfinite(x).all()
        xs.append(x);ys.append(label)
        records.append(dict(step=step,label=label,inverse_delta=inverse_delta,candidates=values,
            native_inverse_hit=bool(hits),feature_max_error=error))
    final=state.to_graph(ref);final.validate()
    assert all(np.array_equal(final.family_degrees(f),ref.family_degrees(f)) for f in ref.family_specs)
    return dict(x=np.stack(xs),y=np.array(ys,dtype=np.int64),ids=np.full(len(ys),tid,dtype=np.int64)),dict(
        kind=kind,trajectory_id=tid,closure_biased=close,steps=steps,records=records,path=path,
        final_graph=b.graph_payload(final),forward_triangle_delta_sum=sum(r['forward_delta'] for r in path),
        seconds=time.monotonic()-start,max_feature_error=max_error)


def verify():
    for p,h in b.read(OUT/'manifest.json')['files'].items():assert b.sha(p)==h,p


def coverage(z):
    x,y=z['x'],z['y'];d=np.rint(x[np.arange(len(y)),y,43].astype(float)*np.sqrt(1565))
    return dict(rows=len(y),negative=int((d<0).sum()),zero=int((d==0).sum()),positive=int((d>0).sum()))


def main():
    b.setup();OUT.mkdir(parents=True,exist_ok=False)
    files=[*Path(__file__).parent.glob('*.py'),Path(__file__).with_name('PROTOCOL.md'),Path(a.__file__),Path(b.__file__),Path(metrics.__file__),
        b.RUN/'data/pubmed/graph.json',*b.FROZEN.rglob('*.py')]
    put('manifest.json',dict(files={str(p):b.sha(p) for p in files},train_ids=list(range(8)),validation_ids=list(range(8,12)),
        unique_trajectories=18,unique_rows=576,logical_rows=768,scientific_forward_steps=2304))
    engineering=[]
    for kind in ['U','B']:
        z,r=trajectory(kind,98,8);put(f'engineering/{kind}.json',r)
        np.savez_compressed(OUT/f'engineering/{kind}.npz',**z);engineering.append(r)
    bound=2*max(r['seconds']/8 for r in engineering)*2304+30
    put('cache_gate.json',dict(passed=bound<=600,conservative_seconds=bound,engineering_forward_steps=16,
        engineering_inverse_roundtrips=4,max_feature_error=max(r['max_feature_error'] for r in engineering)))
    assert bound<=600,bound
    start=time.monotonic();banks={};reports=[]
    for tid in range(12):
        for kind in ['U','B']:
            if kind=='B' and tid%2:
                banks[(kind,tid)]=banks[('U',tid)]
                continue
            assert time.monotonic()-start<600
            z,r=trajectory(kind,tid)
            dest=OUT/f'cache/{kind}_{tid}';dest.mkdir(parents=True,exist_ok=False)
            np.savez_compressed(dest/'data.npz',**z)
            put(Path('cache')/f'{kind}_{tid}'/'record.json',r)
            banks[(kind,tid)]=z;reports.append(dict(kind=kind,trajectory_id=tid,seconds=r['seconds'],
                final_triangle_delta=r['forward_triangle_delta_sum'],**coverage(z)))
            print(kind,tid,coverage(z),round(time.monotonic()-start,2),flush=True)
    for kind in ['U','B']:
        train={k:np.concatenate([banks[(kind,tid)][k] for tid in range(8)]) for k in ['x','y','ids']}
        np.savez_compressed(OUT/f'train_{kind}.npz',**train)
    keys=[('U',tid) for tid in range(8,12)]+[('B',tid) for tid in [8,10]]
    val={k:np.concatenate([banks[key][k] for key in keys]) for k in ['x','y','ids']}
    val['origin']=np.concatenate([np.full(32,0 if key[0]=='U' else 1,dtype=np.int64) for key in keys])
    np.savez_compressed(OUT/'validation.npz',**val)
    verify()
    put('cache_completion.json',dict(completed=True,unique_trajectories=18,logical_rows=768,unique_rows=576,
        scientific_forward_steps=2304,training={kind:coverage(np.load(OUT/f'train_{kind}.npz')) for kind in ['U','B']},
        validation=coverage(val),reports=reports,seconds=time.monotonic()-start,neural_calls=0,
        training_updates=0,new_surrogate_generation_chains=0))


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        if OUT.exists():put('cache_failure.json',dict(type=type(exc).__name__,message=str(exc)))
        raise
