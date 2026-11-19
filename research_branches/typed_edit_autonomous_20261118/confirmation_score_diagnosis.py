"""Read-only loss and coefficient decomposition of the six frozen fits."""
import confirmation_fit as fit
import confirmation_common as contract
from round01 import t,b,np,torch,time,Path,ROOT,SRC
OUT=ROOT/'results/typed_edit_autonomous_20261118/confirmation_v1/score_diagnosis'

def main():
    b.setup();fit.source_check(fit.OUT);fit.source_check(fit.CACHE)
    OUT.mkdir(parents=True,exist_ok=False)
    files=[Path(__file__),SRC/'CONFIRMATION_SCORE_DIAGNOSIS.md',fit.OUT/'manifest.json',fit.OUT/'acceptance.json',fit.OUT/'summary.json']
    files+=list(fit.OUT.glob('fits/*/*/checkpoint.pt'))+list(fit.OUT.glob('inputs/*.npz'))
    files+=list(fit.CACHE.glob('*/schema.json'))+list(fit.CACHE.glob('*/repeat*/normalization.json'))+list(fit.CACHE.glob('*/repeat*/new_train.npz'))
    contract.bind(OUT,files,new_fits=0,new_graphs=0,changes_to_confirmation=False)
    tic=time.monotonic();summaries=[];coordinates=[]
    for ds in fit.DATASETS:
        ref=b.graph(ds);rc=t.metrics.counts(ref);cols=b.read(fit.CACHE/ds/'schema.json')['columns']
        reference=np.array([rc[a][k] for a,k in cols]);original=np.load(b.RUN/f'cache/{ds}/data.npz');keep=original['ids']<30
        for repeat in fit.REPEATS:
            assert time.monotonic()-tic<180
            x,y,weight,phi=fit.data(ds,repeat);m=fit.model(ds,repeat)
            cp=torch.load(fit.OUT/f'fits/{ds}/{repeat}/checkpoint.pt',map_location='cpu',weights_only=False)
            w=cp['state_dict']['weights'].numpy();qweight=np.r_[np.zeros(m.k),np.ones(m.k)]
            noop=original['x'][keep][np.arange(3000),y[:3000],0]>.5
            for arm,coeff in [('F',w),('Q',qweight)]:
                logits=phi@coeff;maximum=logits.max(1)
                ce=maximum+np.log(np.exp(logits-maximum[:,None]).sum(1))-logits[np.arange(len(y)),y]
                objective,_=fit.solver.objective(coeff,phi,y,weight)
                summaries.append(dict(dataset=ds,repeat=repeat,arm=arm,objective=objective,weighted_CE=float(weight@ce),old_all_CE=float(ce[:3000].mean()),old_noop_CE=float(ce[:3000][noop].mean()),new_CE=float(ce[3000:].mean()),old_accuracy=float((logits[:3000].argmax(1)==y[:3000]).mean()),new_accuracy=float((logits[3000:].argmax(1)==y[3000:]).mean())))
            mu=m.mu.numpy();sd=m.sd.numpy()
            for j,(family,key) in enumerate(cols):
                linear=float(w[j]);quadratic=float(w[m.k+j]);center=float(mu[j]-linear*sd[j]/(2*quadratic)) if quadratic>1e-10 else None
                coordinates.append(dict(dataset=ds,repeat=repeat,family=family,key=key,reference_count=int(reference[j]),training_mean=float(mu[j]),training_sd=float(sd[j]),linear=linear,quadratic=quadratic,continuous_stationary_count=center,negative_stationary_count=center is not None and center<0,graph_feasibility_not_implied=True))
    contract.verify(OUT);contract.put(OUT,'summary.json',dict(losses=summaries,coordinates=coordinates,changes_to_confirmation=False,not_a_generation_comparison=True))
    contract.put(OUT,'acceptance.json',dict(passed=True,frozen_fits=6,loss_rows=12,coordinates=len(coordinates),new_fits=0,new_graphs=0,seconds=time.monotonic()-tic))
    print('Frozen coefficient diagnosis accepted',len(coordinates),'coordinates',flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        if OUT.exists():contract.put(OUT,'failure.json',dict(type=type(exc).__name__,message=str(exc)))
        raise
