"""Strict stopping for deterministic constrained inverse-CE fitting."""
import round25_fit as prior_fit
import round24_complete as previous
import round20_train as old
from scipy.optimize import minimize
from round01 import t,b,np,torch,time,Path,ROOT,SRC,json
OUT=ROOT/'results/typed_edit_autonomous_20261118/r26_nocache';DATASETS=['pubmed','imdb'];REG=1e-4


def put(name,v):
    f=OUT/name;f.parent.mkdir(parents=True,exist_ok=True);f.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def model(ds):return old.model(ds).double()


def data(ds):
    a,z=old.data(ds);m=model(ds)
    if ds=='imdb':noop=a['target_action']==0
    else:
        original,_,_,_=old.prior.prior.old.train.r11.data();noop=original['x'][np.arange(len(a['y'])),a['y'],0]>.5
    x=np.concatenate([a['x'],z['x']]).astype(np.float64);y=np.concatenate([a['y'],z['y']])
    weight=np.concatenate([np.full(len(a['y']),.25/len(a['y']))+noop*(.25/noop.sum()),np.full(len(z['y']),.5/len(z['y']))]);assert abs(weight.sum()-1)<1e-12
    with torch.no_grad():phi=m.phi(torch.from_numpy(x)).numpy()
    return x,y,weight,phi


def objective(w,phi,y,weight):
    q=phi@w;maximum=q.max(1);exp=np.exp(q-maximum[:,None]);den=exp.sum(1);loss=maximum+np.log(den)-q[np.arange(len(y)),y]
    diff=exp/den[:,None];diff[np.arange(len(y)),y]-=1;diff*=weight[:,None]
    grad=np.einsum('ij,ijk->k',diff,phi,optimize=True)+REG*w
    return float(weight@loss+.5*REG*(w@w)),grad


def pg(w,g):
    result=g.copy();k=len(w)//2;result[k:]=np.where((w[k:]<=1e-8)&(g[k:]>0),0,g[k:]);return float(abs(result).max())


def solve(phi,y,weight):
    calls=0;history=[];tic=time.monotonic()
    def fun(w):
        nonlocal calls
        assert time.monotonic()-tic<600;calls+=1;f,g=objective(w,phi,y,weight);history.append(dict(call=calls,objective=f,projected_gradient_max=pg(w,g)));return f,g
    width=phi.shape[-1];res=minimize(fun,np.zeros(width),method='L-BFGS-B',jac=True,bounds=[(None,None)]*(width//2)+[(0,None)]*(width//2),options=dict(maxiter=400,maxfun=800,ftol=0.,gtol=1e-7))
    f,g=objective(res.x,phi,y,weight)
    report=dict(success=bool(res.success),status=int(res.status),message=str(res.message),iterations=int(res.nit),function_gradient_calls=calls,objective=f,projected_gradient_max=pg(res.x,g),minimum_quadratic=float(res.x[width//2:].min()),seconds=time.monotonic()-tic,history=history)
    return res.x,report


def verify():
    previous.verify();old.verify();prior_fit.verify()
    for f,h in b.read(OUT/'manifest.json')['files'].items():assert b.sha(f)==h,f


def main():
    b.setup();previous.verify();old.verify();prior_fit.verify();assert b.read(previous.OUT/'acceptance.json')['passed'];OUT.mkdir(parents=True,exist_ok=False)
    files=[Path(__file__),SRC/'ROUND25.md',SRC/'ROUND26.md',SRC/'ROUND26_NOCACHE.md',SRC/'round26_fit.py',ROOT/'results/typed_edit_autonomous_20261118/r26_fit/failure.json',prior_fit.OUT/'acceptance.json',prior_fit.OUT/'summary.json',SRC/'potential.py',previous.OUT/'summary.json',previous.OUT/'acceptance.json',old.OUT/'acceptance.json']
    files.extend(old.OUT/f'fits/{ds}/{seed}/checkpoint.pt' for ds in DATASETS for seed in old.SEEDS)
    put('manifest.json',dict(files={str(f):b.sha(f) for f in files},datasets=DATASETS,deterministic_coefficient_fits=2,new_SGD_updates=0,new_original_MLP_fits=0,regularization=REG,solver='L-BFGS-B',maxiter=400,maxfun=800));banks={};audits=[];eng=[];cost=20.
    for ds in DATASETS:
        x,y,weight,phi=data(ds);banks[ds]=(x,y,weight,phi);m=model(ds);w=np.random.default_rng(2026111825).normal(0,.1,phi.shape[-1]);w[len(w)//2:]=abs(w[len(w)//2:])
        f,g=objective(w,phi,y,weight)
        with torch.no_grad():m.weights.copy_(torch.from_numpy(w))
        q=m(torch.from_numpy(x));loss=(torch.nn.functional.cross_entropy(q,torch.from_numpy(y),reduction='none')*torch.from_numpy(weight)).sum()+.5*REG*m.weights.square().sum();loss.backward()
        loss_error=abs(float(loss)-f);grad_error=float(abs(m.weights.grad.numpy()-g).max());assert loss_error<=1e-9 and grad_error<=1e-9
        errors=[]
        for j in np.linspace(0,len(w)-1,8,dtype=int):
            plus=w.copy();minus=w.copy();plus[j]+=1e-5;minus[j]-=1e-5;fd=(objective(plus,phi,y,weight)[0]-objective(minus,phi,y,weight)[0])/2e-5;errors.append(abs(fd-g[j]))
        assert max(errors)<=1e-7
        tic=time.monotonic()
        for _ in range(20):objective(w,phi,y,weight)
        seconds=time.monotonic()-tic;cost+=2*800*seconds/20;eng.append(dict(dataset=ds,loss_error=loss_error,gradient_error=grad_error,finite_difference_error=max(errors),timing_evaluations=20,timing_seconds=seconds))
        for seed in old.SEEDS:
            cp=torch.load(old.OUT/f'fits/{ds}/{seed}/checkpoint.pt',map_location='cpu',weights_only=False);ww=cp['state_dict']['weights'].double().numpy();ff,gg=objective(ww,phi,y,weight)
            audits.append(dict(dataset=ds,seed=seed,objective=ff,projected_gradient_max=pg(ww,gg)))
    put('old_objective_audit.json',audits);put('engineering.json',dict(passed=cost<=600,cost_bound_seconds=cost,checks=eng,new_SGD_updates=0));assert cost<=600;verify();tic=time.monotonic();reports=[]
    for ds in DATASETS:
        x,y,weight,phi=banks[ds];w,rp=solve(phi,y,weight);folder=OUT/f'fits/{ds}';folder.mkdir(parents=True,exist_ok=False);m=model(ds)
        with torch.no_grad():m.weights.copy_(torch.from_numpy(w));q=m(torch.from_numpy(x)).numpy()
        score_error=float(abs(q-phi@w).max());assert score_error<=1e-9
        torch.save(dict(state_dict=m.state_dict(),deterministic=True,optimizer='L-BFGS-B',dtype='float64'),folder/'checkpoint.pt');np.savez_compressed(folder/'training_scores.npz',logits=q,targets=y,weights=weight)
        prior_weights=torch.load(prior_fit.OUT/f'fits/{ds}/checkpoint.pt',map_location='cpu',weights_only=False)['state_dict']['weights'].numpy()
        rp.update(max_coefficient_difference_from_R25=float(abs(w-prior_weights).max()),dataset=ds,score_error=score_error,better_than_all_old=all(rp['objective']<=v['objective']+1e-8 for v in audits if v['dataset']==ds));rp['numerical_gate_passed']=rp['projected_gradient_max']<=1e-5 and rp['better_than_all_old'];reports.append(rp);put(f'fits/{ds}/report.json',rp)
        print('full objective fitted',ds,rp['iterations'],rp['objective'],rp['projected_gradient_max'],rp['numerical_gate_passed'],flush=True)
    put('completion.json',dict(completed=True,new_from_scratch_fits=2,deterministic_coefficient_fits=2,new_SGD_updates=0,new_original_MLP_fits=0,function_gradient_calls=sum(v['function_gradient_calls'] for v in reports),optimizer_iterations=sum(v['iterations'] for v in reports),seconds=time.monotonic()-tic));tic=time.monotonic();replays=[]
    for ds in DATASETS:
        x,y,weight,phi=data(ds);w,rp=solve(phi,y,weight);cp=torch.load(OUT/f'fits/{ds}/checkpoint.pt',map_location='cpu',weights_only=False);wanted=cp['state_dict']['weights'].numpy();error=float(abs(w-wanted).max());original=next(v for v in reports if v['dataset']==ds)
        assert error<=1e-10 and abs(rp['objective']-original['objective'])<=1e-10 and rp['function_gradient_calls']==original['function_gradient_calls'];replays.append(dict(dataset=ds,coefficient_error=error,function_gradient_calls=rp['function_gradient_calls'],objective_error=abs(rp['objective']-original['objective'])))
        print('full objective replayed',ds,error,flush=True)
    verify();put('summary.json',dict(decisions={v['dataset']:v['numerical_gate_passed'] for v in reports},passed=all(v['numerical_gate_passed'] for v in reports),deterministic=True,training_seed_variance_not_estimated=True))
    put('acceptance.json',dict(passed=True,replays=replays,replay_function_gradient_calls=sum(v['function_gradient_calls'] for v in replays),seconds=time.monotonic()-tic))


if __name__=='__main__':
    try:main()
    except BaseException as exc:put('failure.json',dict(type=type(exc).__name__,message=str(exc)));raise
