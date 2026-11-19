"""Paired coefficient fits with targeted versus equal-size ordinary augmentation."""
import argparse
import acm_scaled_solver as scaled
import round26_nocache as standard
import confirmation_fit as original
import crossdomain_augmentation_diagnosis as diagnosis
import confirmation_common as contract
from potential import Potential
from round01 import b,np,torch,time,Path,ROOT,SRC

BASE=ROOT/'results/typed_edit_crossdomain_augmentation_20261118'

def model(ds,repeat):
    a=diagnosis.assets(ds,repeat);schema=b.read(a['schema']);state=torch.load(a['checkpoint'],map_location='cpu',weights_only=False)['state_dict']
    m=Potential(schema['n'],schema['columns'],state['mu'].numpy(),state['sd'].numpy()).double();m.load_state_dict(state);return m

def data(ds,repeat,bank,kind):
    a=diagnosis.assets(ds,repeat);old=np.load(a['old']);new=np.load(a['cache']/'new_train.npz');aug=np.load(BASE/f'cache/{ds}/bank{bank}/{kind}/augmentation.npz')
    orig=np.load(b.RUN/f'cache/{ds}/data.npz');keep=orig['ids']<30;assert keep.sum()==len(old['y'])==3000
    assert np.array_equal(old['ids'],orig['ids'][keep]) and np.array_equal(old['y'],orig['y'][keep])
    noop=orig['x'][keep][np.arange(3000),old['y'],0]>.5;assert noop.any() and len(aug['y'])==256
    x=np.concatenate([old['x'],new['x'],aug['x']]).astype(np.float64);y=np.concatenate([old['y'],new['y'],aug['y']])
    weight=np.r_[np.full(3000,.25/3000)+noop*(.25/noop.sum()),np.full(len(new['y'])+256,.5/(len(new['y'])+256))];assert abs(weight.sum()-1)<1e-12
    with torch.no_grad():phi=model(ds,repeat).phi(torch.from_numpy(x)).numpy()
    return x,y,weight,phi

def main(ds,repeat,bank):
    b.setup();cache=BASE/f'cache/{ds}/bank{bank}';original.source_check(cache);assert b.read(cache/'summary.json')['passed']
    a=diagnosis.assets(ds,repeat);out=BASE/f'confirmation/fit/{ds}/bank{bank}';out.mkdir(parents=True,exist_ok=False)
    files=[Path(__file__),SRC/'CROSSDOMAIN_AUGMENTATION_CONFIRMATION.md',a['checkpoint'],a['old'],a['cache']/'new_train.npz',a['cache']/'normalization.json',b.RUN/f'cache/{ds}/data.npz',cache/'manifest.json',cache/'acceptance.json']
    files += [cache/f'{kind}/augmentation.npz' for kind in ['targeted','ordinary']]

    contract.bind(out,files,dataset=ds,base_repeat=repeat,bank=bank,new_coefficient_fits=2,new_original_MLP_fits=0,normalization_changed=False,Q_changed=False)
    solver=scaled if ds=='acm' else standard;reports=[];output={}
    for kind in ['targeted','ordinary']:
        x,y,weight,phi=data(ds,repeat,bank,kind);m=model(ds,repeat);old=m.weights.detach().numpy().copy();k=m.k
        w=np.random.default_rng(2026111893).normal(0,.1,2*k);w[k:]=abs(w[k:]);value,grad=solver.objective(w,phi,y,weight)
        with torch.no_grad():m.weights.copy_(torch.from_numpy(w))
        logits=m(torch.from_numpy(x));loss=(torch.nn.functional.cross_entropy(logits,torch.from_numpy(y),reduction='none')*torch.from_numpy(weight)).sum()+.5*solver.REG*m.weights.square().sum();loss.backward()
        le=abs(float(loss)-value);ge=float(abs(m.weights.grad.numpy()-grad).max());assert le<=1e-9 and ge<=1e-9
        fd=[]
        for j in np.linspace(0,2*k-1,8,dtype=int):
            plus=w.copy();minus=w.copy();plus[j]+=1e-6;minus[j]-=1e-6;fd.append(abs((solver.objective(plus,phi,y,weight)[0]-solver.objective(minus,phi,y,weight)[0])/2e-6-grad[j]))
        assert max(fd)<=1e-7;tic=time.monotonic()
        for _ in range(20):solver.objective(w,phi,y,weight)
        cost=20+2*(8000 if ds=='acm' else 800)*(time.monotonic()-tic)/20
        contract.put(out,f'{kind}/engineering.json',dict(passed=cost<=600,cost_bound_seconds=cost,loss_error=le,gradient_error=ge,finite_difference_error=max(fd)))
        assert cost<=600;contract.verify(out);tic=time.monotonic();w,rp=solver.solve(phi,y,weight)
        assert rp['projected_gradient_max']<=1e-5 and time.monotonic()-tic<600
        m=model(ds,repeat)
        with torch.no_grad():m.weights.copy_(torch.from_numpy(w));q=m(torch.from_numpy(x)).numpy()
        error=float(abs(q-phi@w).max());assert error<=1e-9
        base=model(ds,repeat)
        for key in ['mu','sd','delta_scale']:assert torch.equal(base.state_dict()[key],m.state_dict()[key])
        previous,_=solver.objective(old,phi,y,weight);assert rp['objective']<=previous+1e-8
        folder=out/kind;torch.save(dict(state_dict=m.state_dict(),dataset=ds,base_repeat=repeat,augmentation_bank=bank,kind=kind,normalization_changed=False),folder/'checkpoint.pt')
        np.savez_compressed(folder/'training_scores.npz',logits=q,targets=y,weights=weight)
        rp.update(kind=kind,score_error=error,base_on_augmented_objective=previous,scientific_seconds=time.monotonic()-tic);contract.put(out,f'{kind}/report.json',rp)
        tic=time.monotonic();xx,yy,ww,pp=data(ds,repeat,bank,kind);replay,rr=solver.solve(pp,yy,ww)
        err=float(abs(replay-w).max());assert err<=1e-10 and abs(rr['objective']-rp['objective'])<=1e-10 and rr['function_gradient_calls']==rp['function_gradient_calls'] and time.monotonic()-tic<600
        reports.append(dict(kind=kind,projected_gradient_max=rp['projected_gradient_max'],coefficient_replay_error=err,objective=rp['objective']))
        output[str(folder/'checkpoint.pt')]=b.sha(folder/'checkpoint.pt');print(ds,kind,'fit accepted',reports[-1],flush=True)
    contract.verify(out);contract.put(out,'summary.json',dict(passed=True,dataset=ds,base_repeat=repeat,bank=bank,reports=reports,normalization_changed=False,Q_changed=False))
    contract.put(out,'acceptance.json',dict(passed=True,outputs=output,new_coefficient_fits=2,new_original_MLP_fits=0))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset',choices=['pubmed','acm']);p.add_argument('--repeat',type=int,required=True);p.add_argument('--bank',type=int,required=True);args=p.parse_args()
    try:main(args.dataset,args.repeat,args.bank)
    except BaseException as exc:
        out=BASE/f'confirmation/fit/{args.dataset}/bank{args.bank}'
        if out.exists():contract.put(out,'failure.json',dict(type=type(exc).__name__,message=str(exc)))
        raise


