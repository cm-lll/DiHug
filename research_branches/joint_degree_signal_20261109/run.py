"""CPU audit of archived probabilities; no model execution or new graph draws."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,csv,time,hashlib,resource,signal,traceback
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'research_branches/joint_degree_distance_20261109'))
from run import JointDegree,degree
OUT=Path(os.environ.get('SIGNAL_OUTPUT',str(ROOT/'results/joint_degree_signal_20261109')))
OLD=ROOT/'results/hard_structure_response_20261108T182154'
EDGE=ROOT/'results/hard_structure_edge_direction_cpu_20261108'
START=time.monotonic(); BIND={}
def bind(p):
    p=Path(p);BIND[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return p
def read(p):return json.loads(bind(p).read_text())
def save(n,x):
    (OUT/(n+'.json')).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    if isinstance(x,list) and x and isinstance(x[0],dict):
        with (OUT/(n+'.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,list(x[0]));w.writeheader();w.writerows(x)
def layout(x):
    if isinstance(x,dict):return {str(k):layout(v) for k,v in x.items()}
    if torch.is_tensor(x) or isinstance(x,np.ndarray):return dict(shape=list(x.shape),dtype=str(x.dtype))
    if isinstance(x,(list,tuple)):return dict(container=type(x).__name__,length=len(x),first=layout(x[0]) if x else None)
    return type(x).__name__
def cosine(a,b):
    den=np.linalg.norm(a)*np.linalg.norm(b);return float(a@b/den) if den else None
def main():
    OUT.mkdir(exist_ok=False);signal.alarm(600)
    save('protocol',dict(scope='archived CPU only; no score/weight/model changes',conditional_states=24,probability_panels=12,selected_slots=2765,
        formula='g_e=p_e(1-p_e)[L(B_-e,1)-L(B_-e,0)]; sample conditional derivative, not exact expectation or parameter gradient',
        selection='reuse all previous relation/presence-stratified slots; no new selection; summary restricted to selected coordinates',
        comparisons=['previous explicit TV score','geometric joint-degree substitution'],
        bounds=dict(seconds=600,threads=1,RSS_MiB=2048,GPU=0),
        limitations='IMDB0411/PubMed0422 historical CE/PS parents after CE continuation, not RC-trained models or PubMed0421 final graph. No fitted lambda; no AdamW or parameter claim. Two archived hard draws not a variance convergence test.'))
    bind(Path(__file__));bind(ROOT/'research_branches/joint_degree_distance_20261109/run.py')
    for p in ['research_branches/rooted_connection_20261109/runtime/objective.py','research_branches/rooted_connection_20261109/runtime/slot_update.py','research_branches/hard_structure_response_20261108/run.py']:bind(ROOT/p)
    # Inspect every archived chain state in the known-case directory, CPU only.
    sample=ROOT/'results/cn_multiseed_20261104T042546/run/PubMed_2026110421_A_late/ema_0.99/sample_2026110232'
    states=[]
    for p in sorted(sample.glob('*.pt')):
        expected=read(p.with_suffix('.sha256.json'))['sha256'];bind(p);assert BIND[str(p)]==expected
        z=torch.load(p,map_location='cpu',weights_only=False)
        states.append(dict(path=str(p),layout=layout(z)))
    save('case_checkpoint_layouts',states)
    save('case_file_inventory',[dict(name=p.name,bytes=p.stat().st_size) for p in sorted(sample.iterdir()) if p.is_file()])
    for name in ['binding.json','identity.json','summary.json']:read(sample/name)
    for name in ['steps.jsonl','trajectory.jsonl']:
        lines=[json.loads(s) for s in bind(sample/name).read_text().splitlines() if s.strip()]
        save(name.replace('.jsonl','_layout'),dict(rows=len(lines),keys=sorted(set().union(*(r.keys() for r in lines)))))
    prev=read(ROOT/'results/explicit_structure_cpu_20261109/binding.json')
    identity=read(ROOT/'results/identity_structure_audit_20261109/asset_source_binding.json')
    def checked(p):
        p=bind(p);h=prev.get(str(p),identity.get(str(p)));assert h==BIND[str(p)],str(p);return p
    rows=read(ROOT/'results/explicit_structure_cpu_20261109/conditional_edges.json')
    assets=read(OLD/'assets.json');probes=read(OLD/'probes.json')
    groups=defaultdict(list)
    for r in rows:groups[(r['dataset'],r['parent_history'],r['t'],r['draw'])].append(r)
    models={};inputs={};out=[];panels=[];vectors={};bindings=[]
    for ds in ['IMDB','PubMed']:
        with np.load(checked(ROOT/f'results/rooted_connection_20261109T015142/{ds}_cpu_cost_panel.npz')) as z:models[ds]=JointDegree(z)
        inputs[ds]={int(t):(seed,inp) for t,seed,inp in torch.load(checked(OLD/f'{ds}_fixed_inputs.pt'),map_location='cpu',weights_only=False)['inputs']}
    for (ds,parent,t,draw),rr in sorted(groups.items()):
        assert time.monotonic()-START<600
        m=models[ds];noise_seed,inp=inputs[ds][t];q,f,y=map(np.asarray,(inp['query'],inp['f'],inp['labels']))
        # Saved prediction panels retain their query order, whereas final-graph
        # score fixtures use canonical order. Verify full identity before adapting.
        n=len(m.types);keys=q[0]*n+q[1];basekeys=m.q[0]*n+m.q[1]
        ix=np.argsort(keys);jx=np.argsort(basekeys)
        assert np.array_equal(keys[ix],basekeys[jx])
        assert np.array_equal(f[ix],m.f[jx]) and np.array_equal(y[ix],m.refbits[jx])
        m=JointDegree(dict(query=q,families=f,reference=y,types=m.types))
        tag=f'{ds}_{parent}_CE_t{t}'
        with np.load(checked(OLD/f'{tag}.npz')) as z:p=z['probability'].astype(float)
        with np.load(checked(EDGE/f'{tag}_draw{draw}_flips.npz')) as z:b=(1-z['flip_sign'])/2
        assert p.shape==y.shape and np.isfinite(p).all() and np.all((p>=0)&(p<=1))
        pr=next(r for r in probes if r['condition']==f'{ds}_{parent}' and r['arm']=='CE' and r['t']==t)
        asset=next(r for r in assets if r['condition']==f'{ds}_{parent}')
        ai=next(r for r in asset['fixed_inputs'] if r['t']==t)
        assert pr['input_sha256']==ai['sha256'] and pr['noise_seed']==noise_seed==ai['seed']
        bindings.append(dict(panel=tag,draw=draw,train_seed=pr['seed'],updates=pr['updates'],readout=pr['readout'],input_hash=pr['input_sha256']))
        d=degree(q,f,b,len(m.types));base,_=m.terms(d);g0=[];g1=[];ce=[];selected=[]
        for r in rr:
            e=r['slot'];assert int(b[e])==r['present'] and int(f[e])==r['relation']
            val,_=m.terms(m.flip_degree(d,e,b[e]));dt=val['TV']-base['TV'];dj=val['joint']-base['joint']
            assert abs(dt-r['joint_degree_TV_change'])<1e-10
            a=r['explicit_change'];c=a+(dj-dt)/8
            factor=float(p[e]*(1-p[e]));s=1-2*b[e]
            old=factor*a*s;new=factor*c*s;cg=float((p[e]-y[e])/(192*191/2))
            g0.append(old);g1.append(new);ce.append(cg);selected.append(e)
            out.append(dict(dataset=ds,parent=parent,train_seed=pr['seed'],t=t,draw=draw,slot=e,relation=int(f[e]),label=int(y[e]),present=int(b[e]),
                probability=float(p[e]),probability_factor=factor,old_flip_delta=a,new_flip_delta=c,old_logit_derivative=old,new_logit_derivative=new,CE_logit_derivative=cg,
                triangle_error_change=r['triangle_error_change'],hub_error_change=r['hub_error_change'],isolates_error_change=r['isolates_error_change'],
                old_CE_opposed=bool(old*cg<0),new_CE_opposed=bool(new*cg<0)))
        x=np.array(g0);z=np.array(g1);c=np.array(ce);vectors[(tag,draw)]={e:v for e,v in zip(selected,z)}
        now=out[-len(rr):]
        for method,key,g in [('old','old_flip_delta',x),('new','new_flip_delta',z)]:
            good=[r for r in now if r[key]<-1e-12];total=float(abs(g).sum())
            bad=[r for r in good if r['triangle_error_change']>1e-12]
            field=method+'_logit_derivative'
            panels.append(dict(dataset=ds,parent=parent,train_seed=pr['seed'],t=t,draw=draw,method=method,selected_slots=len(rr),rewarded=len(good),
                triangle_worse=len(bad),hub_worse=sum(r['hub_error_change']>1e-12 for r in good),isolates_worse=sum(r['isolates_error_change']>0 for r in good),
                selected_gradient_norm=float(np.linalg.norm(g)),selected_cosine_CE=cosine(g,c),CE_opposed=int((g*c<0).sum()),
                rewarded_triangle_cost_abs_derivative_sum=float(sum(abs(r[field]) for r in bad)),
                selected_abs_derivative_sum=total,probability_zero_or_one=int(sum(r['probability_factor']==0 for r in now))))
    assert len(out)==2765 and len(groups)==24
    save('slot_feedback',out);save('panel_summary',panels);save('conditional_identity',bindings)
    pairs=[]
    for tag in sorted(set(k[0] for k in vectors)):
        a=vectors[(tag,0)];b=vectors[(tag,1)];ix=sorted(a.keys()&b.keys())
        x=np.array([a[e] for e in ix]);y=np.array([b[e] for e in ix])
        pairs.append(dict(panel=tag,common_slots=len(ix),cosine=cosine(x,y),sign_disagreements=int((x*y<0).sum()),limitation='intersection of prior stratified selections, not full vector or variance estimate'))
    save('draw_overlap',pairs)
    assert not torch.cuda.is_initialized()
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024<2048
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in BIND.items())
    save('binding',BIND);save('receipt',dict(status='complete',seconds=time.monotonic()-START,RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,slots=len(out),models=0,GPU_hours=0,updates=0,new_graphs=0,assets_unchanged=True))
    print(json.dumps(json.loads((OUT/'receipt.json').read_text())))
if __name__=='__main__':
    try:main()
    except BaseException as exc:
        if OUT.exists():save('failure',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise
