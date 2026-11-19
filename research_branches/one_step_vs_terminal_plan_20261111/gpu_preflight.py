"""Frozen-policy engineering/cost panel. No optimizer, updates, or new chains.

S1: one clean draw with no baseline. S16: each draw's LOO cost advantage.
T_eager: terminal graph clean-head score with four-trajectory LOO advantage;
this is NOT an unbiased terminal-policy gradient. Old paths are reused only
at their original policy, and all three VJPs share each saved Xt forward.
"""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import sys, json, time, hashlib, importlib.util, traceback, random
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT/'results/terminal_structure_real_20261109T233600'
def module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT/path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m

visible = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
heat = module('preflight_heat', 'research_branches/typed_heat_audit_20261111/run.py')
joint = module('preflight_joint', 'research_branches/typed_heat_degree_audit_20261111/run.py')
# Old CPU reference modules deliberately hide CUDA; undo only this import side effect.
os.environ['CUDA_VISIBLE_DEVICES'] = visible
w = module('preflight_production', 'research_branches/terminal_structure_real_20261109/worker.py')
torch = w.torch

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path, value):
    path = Path(path); tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)); tmp.replace(path)
def equal(a,b):
    if isinstance(a, torch.Tensor): return torch.equal(a,b)
    if isinstance(a, np.ndarray): return np.array_equal(a,b)
    if isinstance(a, dict): return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)): return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b
def rng():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())

class Score:
    def __init__(self, ds, case, query, fam, labels):
        self.types=case['space'].types; self.n=len(self.types); self.r=case['n_families']
        self.q=query; self.f=fam; self.ref=labels
        self.s=json.loads((ROOT/'results/typed_heat_degree_audit_20261111/reference_scaling.json').read_text())[ds]
        self.h,self.d=self.features(labels)
        self.target=np.bincount(fam,weights=labels,minlength=self.r)
    def features(self,b):
        h=heat.heat(heat.adj(self.n,self.q[:,b.astype(bool)].T),self.types)['feat']
        d=joint.degrees(b,self.q,self.f,self.n,self.r).astype(float)
        for typ in np.unique(self.types):
            sel=self.types==typ
            d[sel]=np.log1p(d[sel])/self.s['log_degree_scales'][str(typ)]/np.sqrt(self.r)
        return h,d
    def __call__(self,b):
        h,d=self.features(b); terms=[]
        for typ in np.unique(self.types):
            sel=self.types==typ; hb=self.s['heat_bandwidth_squared'][str(typ)]; db=self.s['degree_bandwidth_squared'][str(typ)]
            def kernel(ha,da,hb_,db_):
                return joint.kernel(ha,hb_,hb)*joint.kernel(da,db_,db)
            terms.append(joint.mmd(kernel(h[sel],d[sel],h[sel],d[sel]),
                kernel(self.h[sel],self.d[sel],self.h[sel],self.d[sel]),
                kernel(h[sel],d[sel],self.h[sel],self.d[sel])))
        count=np.bincount(self.f,weights=b,minlength=self.r)
        cc=float(np.mean((count-self.target)**2/np.maximum(1,self.target)**2))
        return float(np.mean(terms)),cc,terms

def main():
    out=Path(sys.argv[1]); start=time.time(); protocol=json.loads((out/'protocol.json').read_text())
    rows=[]; calls=0; backwards=0
    def check():
        if time.time()>=protocol['stop_epoch']: raise TimeoutError('15 minute shared window')
    try:
        torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest')
        assert torch.cuda.is_available()
        old=json.loads((OLD/'protocol.json').read_text())
        for p,h in protocol['bindings'].items(): assert sha(p)==h, p
        write(out/'environment.json',w.common.environment('cuda'))
        for ds in ['IMDB','PubMed']:
            check(); row=old['models'][ds]
            z=torch.load(row['readout'],map_location='cpu'); cfg=z['binding']['config']
            assert z['updates']==3000 and z['readout']=='ema_0.99' and z['train_seed']==row['train_seed']
            data=np.load(row['data']); edges=data['edges']; edges=edges.T if edges.shape[0]!=2 else edges
            case=w.common.make_case(data['types'],edges,tuple(map(tuple,data['schema'])))
            model=w.StructuralDenoiser(case,cfg,'A'); model.load_state_dict(z['model'],strict=True); model=model.cuda().eval()
            initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; del z
            params=list(model.parameters())
            ids,fam,query=w.common.plan_for(case,old['sample_seeds'][0],1).block(0)
            all_f=np.repeat(np.arange(case['n_families']),case['space'].counts)
            assert len(ids)==int(np.repeat(case['active'],case['space'].counts).sum()) and len(np.unique(ids))==len(ids)
            assert np.array_equal(all_f[ids],fam)
            # Derive reference from the actual slot codec, not presumed array order.
            ref=np.zeros(case['space'].total,dtype=np.uint8)
            rf,rl=case['space'].encode(edges); ref[case['space'].offsets[rf]+rl]=1
            scorer=Score(ds,case,query,fam,ref[ids])
            terminal=[]; costs=[]
            for ss in old['sample_seeds']:
                g=np.load(OLD/ds/str(ss)/'graph.npz'); e=g['edges']; e=e.T if e.shape[0]!=2 else e
                b=np.zeros(case['space'].total,dtype=np.uint8); gf,gl=case['space'].encode(e); b[case['space'].offsets[gf]+gl]=1
                assert not np.any(b[~np.repeat(case['active'],case['space'].counts)])
                c=scorer(b[ids]); terminal.append(b[ids]); costs.append(c[0]+c[1])
            costs=np.asarray(costs); adv=(4*costs-costs.sum())/3
            write(out/(ds+'_terminal_costs.json'),dict(costs=costs.tolist(),LOO_cost_advantages=adv.tolist()))
            for chain_idx,ss in enumerate(old['sample_seeds']):
                for path in sorted((OLD/ds/str(ss)).glob('step_*.npz')):
                    check(); tick=time.perf_counter(); zz=np.load(path); t=int(path.stem.split('_')[1])
                    ee,ff=w.common.state_edges(case,zz['xt']); before_rng=rng()
                    logits,p=w.runtime.predict(model,case['space'].types,ee,ff,query,fam,t/1000,details=True)
                    torch.cuda.synchronize(); forward_s=time.perf_counter()-tick; calls+=1
                    assert logits.shape==(len(ids),2) and torch.isfinite(logits).all()
                    lp=torch.log_softmax(logits.double(),dim=-1); prob=lp[:,1].exp().detach().cpu().numpy()
                    p_error=float(np.max(np.abs(prob-zz['p0'][ids])))
                    generator=np.random.default_rng(np.random.SeedSequence([2026111301,row['train_seed'],ss,t]))
                    bits=(generator.random((16,len(ids)))<prob).astype(np.uint8)
                    tick=time.perf_counter(); scored=[scorer(b) for b in bits]
                    c=np.array([x[0]+x[1] for x in scored]); score_s=time.perf_counter()-tick
                    a=(16*c-c.sum())/15
                    targets=torch.as_tensor(bits,device='cuda',dtype=torch.long)
                    loglik=torch.where(targets.bool(),lp[:,1],lp[:,0]).sum(1)
                    term=torch.as_tensor(terminal[chain_idx],device='cuda',dtype=torch.bool)
                    losses=[c[0]*loglik[0],(torch.as_tensor(a,device='cuda')*loglik).mean(),
                            float(adv[chain_idx])*torch.where(term,lp[:,1],lp[:,0]).sum()]
                    # Minimize cost; each graph has its own advantage. No winner selection.
                    # Terminal eager coefficients are a biased surrogate, NOT transition scores.
                    norms=[]; times=[]; vectors=[]
                    for k,loss in enumerate(losses):
                        check(); tick=time.perf_counter()
                        grads=torch.autograd.grad(loss,params,retain_graph=k<2,allow_unused=True); backwards+=1
                        flat=torch.cat([(torch.zeros_like(v) if g is None else g).detach().reshape(-1) for v,g in zip(params,grads)])
                        assert torch.isfinite(flat).all(); torch.cuda.synchronize()
                        norms.append(float(flat.norm())); times.append(time.perf_counter()-tick); vectors.append(flat.cpu())
                    assert equal(before_rng,rng()), 'forward/VJP polluted global RNG'
                    assert all(p.grad is None for p in params)
                    cosine={}
                    for i,j in [(0,1),(0,2),(1,2)]:
                        denom=float(vectors[i].norm()*vectors[j].norm())
                        cosine[f'{i}_{j}']=float(vectors[i]@vectors[j])/denom if denom else None
                    rec=dict(dataset=ds,train_seed=row['train_seed'],sample_seed=ss,t=t,forward_seconds=forward_s,
                        score16_seconds=score_s,VJP_seconds=times,gradient_norms=norms,cosines=cosine,
                        cached_p_max_abs=p_error,joint_mean=float(np.mean([x[0] for x in scored])),count_mean=float(np.mean([x[1] for x in scored])),
                        per_graph_costs=c.tolist(),per_graph_LOO_cost_advantages=a.tolist(),
                        type_joint_costs=[x[2] for x in scored],active_slots=len(ids),
                        unique_clean_draws=len(np.unique(bits,axis=0)))
                    rows.append(rec);write(out/'rows.json',rows)
                    print(json.dumps(dict(completed=len(rows),total=64,dataset=ds,t=t,elapsed=time.time()-start)),flush=True)
                    del logits,p,lp,losses,loss,grads,flat,vectors,loglik
            assert all(torch.equal(v.cpu(),initial[k]) for k,v in model.state_dict().items()), 'model mutated'
            del model,params;torch.cuda.empty_cache()
        assert calls==64 and backwards==192
        for p,h in protocol['bindings'].items(): assert sha(p)==h,p
        write(out/'complete.json',dict(status='complete',forwards=calls,VJPs=backwards,updates=0,new_reverse_chains=0,
            conditional_clean_draws=1024,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),model_and_RNG_unchanged=True))
    except BaseException:
        write(out/'failure.json',dict(traceback=traceback.format_exc(),forwards=calls,VJPs=backwards,completed=len(rows)))
        raise
    finally:
        write(out/'worker_receipt.json',dict(start_epoch=start,end_epoch=time.time(),seconds=time.time()-start))

if __name__=='__main__': main()
