"""Frozen F inverse-CE fitting on three independent corruption banks per reference."""
import confirmation_common as contract
import round26_nocache as solver
import round30_deploy as prior
from potential import Potential
from round01 import t,b,np,torch,time,Path,ROOT,SRC,json
OUT=ROOT/'results/typed_edit_autonomous_20261118/confirmation_v1/fit'
CACHE=ROOT/'results/typed_edit_autonomous_20261118/confirmation_v1/cache_accounted_v2'
DATASETS=['imdb','pubmed'];REPEATS=[1,2,3]
put=lambda name,v:contract.put(OUT,name,v)
verify=lambda:contract.verify(OUT)

def source_check(folder):
    for f,h in b.read(folder/'manifest.json')['files'].items():assert b.sha(f)==h,f
    receipt=b.read(folder/'acceptance.json');assert receipt['passed']
    for f,h in receipt.get('outputs',{}).items():assert b.sha(f)==h,f

def model(ds,repeat):
    schema=b.read(CACHE/ds/'schema.json');norm=b.read(CACHE/ds/f'repeat{repeat}/normalization.json')
    # Preserve F's float32 normalization buffers promoted to double, including rounding.
    return Potential(schema['n'],schema['columns'],norm['mean'],norm['sd']).double()

def data(ds,repeat):
    a=np.load(OUT/f'inputs/{ds}_old_train.npz');z=np.load(CACHE/ds/f'repeat{repeat}/new_train.npz');m=model(ds,repeat)
    original=np.load(b.RUN/f'cache/{ds}/data.npz');keep=original['ids']<30
    assert keep.sum()==3000 and len(a['y'])==3000 and len(z['y'])==1280
    assert np.array_equal(original['y'][keep],a['y']) and np.array_equal(original['ids'][keep],a['ids'])
    original_x=original['x'][keep];noop=original_x[np.arange(3000),a['y'],0]>.5
    assert noop.sum()=={'imdb':577,'pubmed':721}[ds]
    x=np.concatenate([a['x'],z['x']]).astype(np.float64);y=np.concatenate([a['y'],z['y']])
    weight=np.concatenate([np.full(3000,.25/3000)+noop*(.25/noop.sum()),np.full(1280,.5/1280)])
    assert abs(weight.sum()-1)<1e-12
    with torch.no_grad():phi=m.phi(torch.from_numpy(x)).numpy()
    return x,y,weight,phi

def main():
    b.setup();prior.verify();source_check(CACHE);OUT.mkdir(parents=True,exist_ok=False)
    files=[Path(__file__),SRC/'CONFIRMATION_V1.md',SRC/'CONFIRMATION_CACHE_ACCOUNTING.md',CACHE/'manifest.json',CACHE/'acceptance.json',prior.OUT/'summary.json']
    files.extend(Path(f) for f in b.read(CACHE/'acceptance.json')['outputs'])
    files.extend(b.RUN/f'cache/{ds}/data.npz' for ds in DATASETS)
    # Materialize the verified historical reconstruction so every later read is bound.
    (OUT/'inputs').mkdir()
    for ds in DATASETS:
        old,_=solver.old.data(ds);f=OUT/f'inputs/{ds}_old_train.npz';np.savez_compressed(f,**old);files.append(f)
    contract.bind(OUT,files,datasets=DATASETS,repeats=REPEATS,coefficient_fits=6,shared_old_training_rows=3000,old_assessment_rows_excluded=200,new_original_MLP_fits=0)
    cost=20.;eng=[];banks={}
    for ds in DATASETS:
        for repeat in REPEATS:
            x,y,weight,phi=data(ds,repeat);banks[ds,repeat]=(x,y,weight,phi);m=model(ds,repeat)
            w=np.random.default_rng(2026111833).normal(0,.1,phi.shape[-1]);w[len(w)//2:]=abs(w[len(w)//2:]);f,g=solver.objective(w,phi,y,weight)
            with torch.no_grad():m.weights.copy_(torch.from_numpy(w))
            q=m(torch.from_numpy(x));loss=(torch.nn.functional.cross_entropy(q,torch.from_numpy(y),reduction='none')*torch.from_numpy(weight)).sum()+.5*solver.REG*m.weights.square().sum();loss.backward()
            le=abs(float(loss)-f);ge=float(abs(m.weights.grad.numpy()-g).max());assert le<=1e-9 and ge<=1e-9
            fd=[]
            for j in np.linspace(0,len(w)-1,8,dtype=int):
                plus=w.copy();minus=w.copy();plus[j]+=1e-5;minus[j]-=1e-5;fd.append(abs((solver.objective(plus,phi,y,weight)[0]-solver.objective(minus,phi,y,weight)[0])/2e-5-g[j]))
            assert max(fd)<=1e-7;tic=time.monotonic()
            for _ in range(20):solver.objective(w,phi,y,weight)
            elapsed=time.monotonic()-tic;cost+=2*800*elapsed/20;eng.append(dict(dataset=ds,repeat=repeat,loss_error=le,gradient_error=ge,finite_difference_error=max(fd),seconds=elapsed))
    put('engineering.json',dict(passed=cost<=600,cost_bound_seconds=cost,rows=eng));assert cost<=600;verify();tic=time.monotonic();reports=[]
    for ds in DATASETS:
        for repeat in REPEATS:
            assert time.monotonic()-tic<600
            x,y,weight,phi=banks[ds,repeat];w,rp=solver.solve(phi,y,weight);m=model(ds,repeat)
            with torch.no_grad():m.weights.copy_(torch.from_numpy(w));q=m(torch.from_numpy(x)).numpy()
            error=float(abs(q-phi@w).max());assert error<=1e-9
            folder=OUT/f'fits/{ds}/{repeat}';folder.mkdir(parents=True,exist_ok=False)
            torch.save(dict(state_dict=m.state_dict(),dataset=ds,corruption_repeat=repeat,deterministic=True,dtype='float64',optimizer='L-BFGS-B'),folder/'checkpoint.pt')
            np.savez_compressed(folder/'training_scores.npz',logits=q,targets=y,weights=weight)
            rp.update(dataset=ds,repeat=repeat,score_error=error,numerical_gate_passed=rp['projected_gradient_max']<=1e-5)
            put(f'fits/{ds}/{repeat}/report.json',rp);reports.append(rp);print('confirmation F fitted',ds,repeat,rp['projected_gradient_max'],flush=True)
    put('completion.json',dict(completed=True,coefficient_fits=6,new_original_MLP_fits=0,function_gradient_calls=sum(v['function_gradient_calls'] for v in reports),seconds=time.monotonic()-tic));tic=time.monotonic();replays=[]
    for rp in reports:
        ds=rp['dataset'];repeat=rp['repeat'];x,y,weight,phi=data(ds,repeat);w,rr=solver.solve(phi,y,weight)
        cp=torch.load(OUT/f'fits/{ds}/{repeat}/checkpoint.pt',map_location='cpu',weights_only=False)
        error=float(abs(w-cp['state_dict']['weights'].numpy()).max());assert error<=1e-10 and abs(rr['objective']-rp['objective'])<=1e-10 and rr['function_gradient_calls']==rp['function_gradient_calls']
        replays.append(dict(dataset=ds,repeat=repeat,coefficient_error=error,function_gradient_calls=rr['function_gradient_calls']))
    put('summary.json',dict(passed=all(v['numerical_gate_passed'] for v in reports),reports=reports,independent_corruption_banks_per_reference=3,independent_full_system_training_runs=0,shared_old_bank=True,shared_first_stage_MLP=True))
    verify();outputs={str(f):b.sha(f) for f in OUT.glob('fits/*/*/checkpoint.pt')}
    put('acceptance.json',dict(passed=True,replays=replays,outputs=outputs,seconds=time.monotonic()-tic))

if __name__=='__main__':
    try:main()
    except BaseException as exc:put('failure.json',dict(type=type(exc).__name__,message=str(exc)));raise
