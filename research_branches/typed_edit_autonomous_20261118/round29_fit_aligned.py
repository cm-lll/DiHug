"""Conditional inverse CE on the deployed switch/no-op support."""
import round28_extend as prior
import round26_nocache as full
from scipy.optimize import minimize
from round01 import t,b,np,torch,time,Path,ROOT,SRC,json
OUT=ROOT/'results/typed_edit_autonomous_20261118/r29_fit_aligned';DATASETS=['pubmed','imdb'];REG=1e-4
model=full.model

def put(name,v):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')

def data(ds):
    old,new=full.old.data(ds);source=np.load(b.RUN/f'cache/{ds}/data.npz');original={k:source[k][source['ids']<30] for k in ['x','y','ids']}
    assert np.array_equal(original['y'],old['y']) and np.array_equal(original['ids'],old['ids'])
    onehot=original['x'][:,:,:5];assert np.isin(onehot,[0.,1.]).all() and np.array_equal(onehot.sum(-1),np.ones(onehot.shape[:2]))
    action=onehot.argmax(-1);labels=action[np.arange(len(old['y'])),old['y']];keep=np.isin(labels,[0,4]);noop=labels[keep]==0;mask=np.isin(action[keep],[0,4]);assert mask[np.arange(keep.sum()),old['y'][keep]].all()
    if ds=='imdb':
        assert np.array_equal(labels,old['target_action']) and (new['target_action']==4).all()
        records=[]
        for f in (ROOT/'results/typed_edit_autonomous_20261118/r17_cache/cache').glob('*/record.json'):records.extend(b.read(f)['records'])
        assert len(records)==len(new['y']) and all(set(r['candidates']['action'])<={0,4} for r in records)
    else:
        _,raw_new,_,_=full.old.prior.prior.old.train.r11.data();aa=raw_new['x'][:,:,:5].argmax(-1)
        assert np.isin(aa,[0,4]).all() and (aa[np.arange(len(new['y'])),new['y']]==4).all()
    x=np.concatenate([old['x'][keep],new['x']]).astype(np.float64);y=np.concatenate([old['y'][keep],new['y']]);mask=np.concatenate([mask,np.ones((len(new['y']),16),dtype=bool)])
    weight=np.concatenate([np.full(keep.sum(),.25/keep.sum())+noop*(.25/noop.sum()),np.full(len(new['y']),.5/len(new['y']))]);assert abs(weight.sum()-1)<1e-12
    m=model(ds)
    with torch.no_grad():phi=m.phi(torch.from_numpy(x)).numpy()
    audit=dict(dataset=ds,old_records=len(keep),eligible_old=int(keep.sum()),old_switch=int((labels==4).sum()),old_noop=int((labels==0).sum()),new_records=len(new['y']),discarded_non_stage_two_target_weight=float(.25*(~keep).mean()),masked_old_candidate_fraction=float((~mask[:keep.sum()]).mean()),minimum_retained_candidates=int(mask.sum(1).min()))
    audit['passed']=audit['discarded_non_stage_two_target_weight']>=.1 and audit['old_switch']>=100 and audit['old_noop']>=100 and audit['minimum_retained_candidates']>=1
    return x,y,weight,phi,mask,audit

def objective(w,phi,y,weight,mask):
    q=np.where(mask,phi@w,-np.inf);maximum=q.max(1);exp=np.exp(q-maximum[:,None]);den=exp.sum(1);loss=maximum+np.log(den)-q[np.arange(len(y)),y]
    diff=exp/den[:,None];diff[np.arange(len(y)),y]-=1;diff*=weight[:,None]
    return float(weight@loss+.5*REG*(w@w)),np.einsum('ij,ijk->k',diff,phi,optimize=True)+REG*w

def solve(phi,y,weight,mask):
    history=[];tic=time.monotonic()
    def fun(w):
        assert time.monotonic()-tic<600;f,g=objective(w,phi,y,weight,mask);history.append(dict(call=len(history)+1,objective=f,projected_gradient_max=full.pg(w,g)));return f,g
    width=phi.shape[-1];res=minimize(fun,np.zeros(width),method='L-BFGS-B',jac=True,bounds=[(None,None)]*(width//2)+[(0,None)]*(width//2),options=dict(maxiter=400,maxfun=800,ftol=0.,gtol=1e-7))
    f,g=objective(res.x,phi,y,weight,mask)
    return res.x,dict(success=bool(res.success),status=int(res.status),message=str(res.message),iterations=int(res.nit),function_gradient_calls=len(history),objective=f,projected_gradient_max=full.pg(res.x,g),minimum_quadratic=float(res.x[width//2:].min()),seconds=time.monotonic()-tic,history=history)

def verify():
    prior.verify()
    for f,h in b.read(OUT/'manifest.json')['files'].items():assert b.sha(f)==h,f

def main():
    b.setup();prior.verify();assert b.read(prior.OUT/'acceptance.json')['passed'];OUT.mkdir(parents=True,exist_ok=False)
    files=[Path(__file__),SRC/'ROUND29.md',SRC/'ROUND29_ALIGNED.md',SRC/'round29_fit.py',ROOT/'results/typed_edit_autonomous_20261118/r29_fit/failure.json',prior.OUT/'acceptance.json',prior.OUT/'summary.json',full.OUT/'acceptance.json',full.OUT/'summary.json']
    files.extend(b.RUN/f'cache/{ds}/data.npz' for ds in DATASETS);files.extend(full.OUT/f'fits/{ds}/checkpoint.pt' for ds in DATASETS)
    put('manifest.json',dict(files={str(f):b.sha(f) for f in files},datasets=DATASETS,deterministic_coefficient_fits=2,parameterization_unchanged=True,new_original_MLP_fits=0));banks={};eng=[];audits=[];cost=20.
    for ds in DATASETS:
        x,y,weight,phi,mask,audit=data(ds);banks[ds]=(x,y,weight,phi,mask);audits.append(audit);put('support_audit.json',audits);assert audit['passed']
        np.savez_compressed(OUT/f'{ds}_support.npz',mask=mask,targets=y,weights=weight);m=model(ds);w=np.random.default_rng(2026111829).normal(0,.1,phi.shape[-1]);w[len(w)//2:]=abs(w[len(w)//2:]);f,g=objective(w,phi,y,weight,mask)
        with torch.no_grad():m.weights.copy_(torch.from_numpy(w))
        q=m(torch.from_numpy(x)).masked_fill(~torch.from_numpy(mask),-torch.inf);loss=(torch.nn.functional.cross_entropy(q,torch.from_numpy(y),reduction='none')*torch.from_numpy(weight)).sum()+.5*REG*m.weights.square().sum();loss.backward()
        loss_error=abs(float(loss)-f);grad_error=float(abs(m.weights.grad.numpy()-g).max());assert loss_error<=1e-9 and grad_error<=1e-9;fd=[]
        for j in np.linspace(0,len(w)-1,8,dtype=int):
            a=w.copy();c=w.copy();a[j]+=1e-5;c[j]-=1e-5;fd.append(abs((objective(a,phi,y,weight,mask)[0]-objective(c,phi,y,weight,mask)[0])/2e-5-g[j]))
        assert max(fd)<=1e-7;tic=time.monotonic()
        for _ in range(20):objective(w,phi,y,weight,mask)
        seconds=time.monotonic()-tic;cost+=2*800*seconds/20;eng.append(dict(dataset=ds,loss_error=loss_error,gradient_error=grad_error,finite_difference_error=max(fd),timing_evaluations=20,timing_seconds=seconds))
    put('engineering.json',dict(passed=cost<=600,cost_bound_seconds=cost,checks=eng));assert cost<=600;verify();tic=time.monotonic();reports=[]
    for ds in DATASETS:
        x,y,weight,phi,mask=banks[ds];w,rp=solve(phi,y,weight,mask);folder=OUT/f'fits/{ds}';folder.mkdir(parents=True,exist_ok=False);m=model(ds)
        with torch.no_grad():m.weights.copy_(torch.from_numpy(w));q=m(torch.from_numpy(x)).numpy()
        assert abs(q-phi@w).max()<=1e-9;torch.save(dict(state_dict=m.state_dict(),deterministic=True,optimizer='L-BFGS-B',dtype='float64',support='switch/noop'),folder/'checkpoint.pt')
        old_w=torch.load(full.OUT/f'fits/{ds}/checkpoint.pt',map_location='cpu',weights_only=False)['state_dict']['weights'].numpy();old_f,old_g=objective(old_w,phi,y,weight,mask)
        rp.update(dataset=ds,F_on_conditional_objective=old_f,F_conditional_projected_gradient=full.pg(old_w,old_g));rp['numerical_gate_passed']=rp['projected_gradient_max']<=1e-5 and rp['objective']<=old_f+1e-8;reports.append(rp);put(f'fits/{ds}/report.json',rp);print('conditional fitted',ds,rp['objective'],rp['projected_gradient_max'],rp['numerical_gate_passed'],flush=True)
    put('completion.json',dict(completed=True,deterministic_coefficient_fits=2,new_SGD_updates=0,new_original_MLP_fits=0,function_gradient_calls=sum(v['function_gradient_calls'] for v in reports),seconds=time.monotonic()-tic));tic=time.monotonic();replays=[]
    for ds in DATASETS:
        x,y,weight,phi,mask,audit=data(ds);w,rp=solve(phi,y,weight,mask);saved=torch.load(OUT/f'fits/{ds}/checkpoint.pt',map_location='cpu',weights_only=False)['state_dict']['weights'].numpy();error=float(abs(w-saved).max());assert error<=1e-10;original=next(v for v in reports if v['dataset']==ds);assert abs(rp['objective']-original['objective'])<=1e-10 and rp['function_gradient_calls']==original['function_gradient_calls'];replays.append(dict(dataset=ds,coefficient_error=error,function_gradient_calls=rp['function_gradient_calls']))
    put('summary.json',dict(decisions={v['dataset']:v['numerical_gate_passed'] for v in reports},passed=all(v['numerical_gate_passed'] for v in reports),training_seed_variance_not_estimated=True,graph_quality_not_evaluated=True));verify();put('acceptance.json',dict(passed=True,replays=replays,seconds=time.monotonic()-tic));print('conditional numeric acceptance',flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as exc:put('failure.json',dict(type=type(exc).__name__,message=str(exc)));raise
