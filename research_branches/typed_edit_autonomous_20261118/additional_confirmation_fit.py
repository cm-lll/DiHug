"""Frozen full-objective coefficient fit for one separately fitted extra domain."""
import argparse
import confirmation_common as contract
import round26_nocache as solver
from potential import Potential
from round01 import t,b,np,torch,time,Path,ROOT,SRC
BASE=ROOT/'results/typed_edit_autonomous_20261118'
OUT=None;REPEAT=None

def source_check(folder):
    for f,h in b.read(folder/'manifest.json')['files'].items():assert b.sha(f)==h,f
    receipt=b.read(folder/'acceptance.json');assert receipt['passed']
    for f,h in receipt.get('outputs',{}).items():assert b.sha(f)==h,f

def model(ds):
    cache=BASE/f'additional_confirmation_v1/cache/{ds}/{REPEAT}'
    schema=b.read(cache/'schema.json');norm=b.read(cache/'normalization.json')
    return Potential(schema['n'],schema['columns'],norm['mean'],norm['sd']).double()

def data(ds):
    cache=BASE/f'additional_confirmation_v1/cache/{ds}/{REPEAT}';old=BASE/f'additional_old_cache/{ds}'
    a=np.load(old/'old_train.npz');z=np.load(cache/'new_train.npz')
    orig=np.load(b.RUN/f'cache/{ds}/data.npz');keep=orig['ids']<30
    assert keep.sum()==3000 and np.array_equal(orig['y'][keep],a['y']) and np.array_equal(orig['ids'][keep],a['ids'])
    x0=orig['x'][keep];noop=x0[np.arange(3000),a['y'],0]>.5;assert noop.any()
    assert len(z['y'])=={'acm':1664,'freebase':1792}[ds]
    x=np.concatenate([a['x'],z['x']]).astype(np.float64);y=np.concatenate([a['y'],z['y']])
    weight=np.concatenate([np.full(3000,.25/3000)+noop*(.25/noop.sum()),np.full(len(z['y']),.5/len(z['y']))])
    assert abs(weight.sum()-1)<1e-12
    with torch.no_grad():phi=model(ds).phi(torch.from_numpy(x)).numpy()
    return x,y,weight,phi,int(noop.sum())

def main(ds,repeat):
    global OUT,REPEAT,solver
    assert repeat in [2,3,4];REPEAT=repeat
    if ds=='acm':
        import acm_scaled_solver as solver
    b.setup();cache=BASE/f'additional_confirmation_v1/cache/{ds}/{REPEAT}';old=BASE/f'additional_old_cache/{ds}'
    source_check(cache);source_check(old)
    OUT=BASE/f'additional_confirmation_v1/fit/{ds}/{REPEAT}';OUT.mkdir(parents=True,exist_ok=False)
    files=[Path(__file__),SRC/'ADDITIONAL_CONFIRMATION_V1.md',b.RUN/f'cache/{ds}/data.npz']
    for parent in [cache,old]:
        files += [parent/'manifest.json',parent/'acceptance.json']
        files += [Path(f) for f in b.read(parent/'acceptance.json')['outputs']]
    contract.bind(OUT,files,dataset=ds,new_coefficient_fits=1,new_original_MLP_fits=0,corruption_repeat=REPEAT)
    x,y,weight,phi,noops=data(ds);m=model(ds)
    w=np.random.default_rng(2026111843).normal(0,.1,phi.shape[-1]);w[len(w)//2:]=abs(w[len(w)//2:])
    loss,grad=solver.objective(w,phi,y,weight)
    with torch.no_grad():m.weights.copy_(torch.from_numpy(w))
    q=m(torch.from_numpy(x));lt=(torch.nn.functional.cross_entropy(q,torch.from_numpy(y),reduction='none')*torch.from_numpy(weight)).sum()+.5*solver.REG*m.weights.square().sum();lt.backward()
    le=abs(float(lt)-loss);ge=float(abs(m.weights.grad.numpy()-grad).max());assert le<=1e-9 and ge<=1e-9
    fd=[]
    for j in np.linspace(0,len(w)-1,8,dtype=int):
        plus=w.copy();minus=w.copy();plus[j]+=1e-5;minus[j]-=1e-5
        fd.append(abs((solver.objective(plus,phi,y,weight)[0]-solver.objective(minus,phi,y,weight)[0])/2e-5-grad[j]))
    assert max(fd)<=1e-7;tic=time.monotonic()
    for _ in range(20):solver.objective(w,phi,y,weight)
    cost=20+2*(8000 if ds=='acm' else 800)*(time.monotonic()-tic)/20
    contract.put(OUT,'engineering.json',dict(passed=cost<=600,cost_bound_seconds=cost,loss_error=le,gradient_error=ge,finite_difference_error=max(fd)))
    assert cost<=600;contract.verify(OUT);tic=time.monotonic();w,rp=solver.solve(phi,y,weight)
    m=model(ds)
    with torch.no_grad():m.weights.copy_(torch.from_numpy(w));q=m(torch.from_numpy(x)).numpy()
    error=float(abs(q-phi@w).max());assert error<=1e-9
    assert time.monotonic()-tic<600
    torch.save(dict(state_dict=m.state_dict(),dataset=ds,corruption_repeat=REPEAT,deterministic=True,dtype='float64',optimizer='diagonal L-BFGS-B' if ds=='acm' else 'L-BFGS-B'),OUT/'checkpoint.pt')
    np.savez_compressed(OUT/'training_scores.npz',logits=q,targets=y,weights=weight)
    rp.update(dataset=ds,repeat=REPEAT,noop_old_rows=noops,score_error=error,numerical_gate_passed=rp['projected_gradient_max']<=1e-5)
    contract.put(OUT,'report.json',rp);contract.put(OUT,'completion.json',dict(completed=True,new_coefficient_fits=1,new_original_MLP_fits=0,seconds=time.monotonic()-tic))
    tic=time.monotonic();xx,yy,ww,pp,nn=data(ds);replay,rr=solver.solve(pp,yy,ww)
    re=float(abs(replay-w).max());assert re<=1e-10 and abs(rr['objective']-rp['objective'])<=1e-10 and rr['function_gradient_calls']==rp['function_gradient_calls']
    assert time.monotonic()-tic<600;contract.verify(OUT)
    contract.put(OUT,'summary.json',dict(passed=rp['numerical_gate_passed'],dataset=ds,coefficient_fits=1,training_seed_sd=None,shared_original_MLP=True,shared_old_training_bank=True))
    contract.put(OUT,'acceptance.json',dict(passed=True,coefficient_replay_error=re,outputs={str(OUT/'checkpoint.pt'):b.sha(OUT/'checkpoint.pt')},seconds=time.monotonic()-tic))
    print('Additional F fitted and replayed',ds,rp['projected_gradient_max'],flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('dataset',choices=['acm','freebase']);parser.add_argument('repeat',type=int,choices=[2,3,4]);args=parser.parse_args()
    try:main(args.dataset,args.repeat)
    except BaseException as exc:
        if OUT is not None:contract.put(OUT,'failure.json',dict(type=type(exc).__name__,message=str(exc)))
        raise
