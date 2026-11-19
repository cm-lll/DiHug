"""Paired coverage confirmation fit; original bank normalization remains fixed."""
import argparse
import confirmation_fit as base
import confirmation_common as contract
from round01 import t,b,np,torch,time,Path,ROOT,SRC
R=ROOT/'results/typed_edit_autonomous_20261118';OUT=None;CACHE=None;REPEAT=None

def model():return base.model('imdb',REPEAT)

def data():
    old=np.load(base.OUT/'inputs/imdb_old_train.npz');z=np.load(base.CACHE/f'imdb/repeat{REPEAT}/new_train.npz');aug=np.load(CACHE/'augmentation.npz')
    original=np.load(b.RUN/'cache/imdb/data.npz');keep=original['ids']<30
    assert len(old['y'])==3000 and len(z['y'])==1280 and len(aug['y'])==256
    assert np.array_equal(old['y'],original['y'][keep]) and np.array_equal(old['ids'],original['ids'][keep])
    noop=original['x'][keep][np.arange(3000),old['y'],0]>.5;assert noop.sum()==577
    x=np.concatenate([old['x'],z['x'],aug['x']]).astype(np.float64);y=np.concatenate([old['y'],z['y'],aug['y']])
    weight=np.r_[np.full(3000,.25/3000)+noop*(.25/noop.sum()),np.full(1536,.5/1536)]
    assert abs(weight.sum()-1)<1e-12
    with torch.no_grad():phi=model().phi(torch.from_numpy(x)).numpy()
    return x,y,weight,phi

def main(repeat):
    global OUT,CACHE,REPEAT
    assert repeat in [1,2,3];REPEAT=repeat
    OUT=R/f'cycle_repair_confirmation/fit/repeat{repeat}';CACHE=R/f'cycle_repair_confirmation/cache/repeat{repeat+1}'
    b.setup();base.source_check(R/'confirmation_v1/deploy');base.source_check(base.OUT);base.source_check(base.CACHE);base.source_check(CACHE)
    assert b.read(CACHE/'summary.json')['passed'];OUT.mkdir(parents=True,exist_ok=False)
    oldcp=base.OUT/f'fits/imdb/{REPEAT}/checkpoint.pt'
    files=[Path(__file__),SRC/'CYCLE_REPAIR_CONFIRMATION.md',oldcp,base.OUT/'inputs/imdb_old_train.npz',base.CACHE/f'imdb/repeat{REPEAT}/new_train.npz',base.CACHE/f'imdb/repeat{REPEAT}/normalization.json',CACHE/'augmentation.npz',CACHE/'manifest.json',CACHE/'acceptance.json',R/'confirmation_v1/deploy/manifest.json',R/'confirmation_v1/deploy/acceptance.json']
    contract.bind(OUT,files,base_bank_repeat=REPEAT,augmentation_repeat=REPEAT+1,new_coefficient_fits=1,new_original_MLP_fits=0,normalization_changed=False,Q_changed=False)
    x,y,weight,phi=data();m=model();w=np.random.default_rng(2026111852).normal(0,.1,phi.shape[-1]);w[m.k:]=abs(w[m.k:])
    value,gradient=base.solver.objective(w,phi,y,weight)
    with torch.no_grad():m.weights.copy_(torch.from_numpy(w))
    logits=m(torch.from_numpy(x));loss=(torch.nn.functional.cross_entropy(logits,torch.from_numpy(y),reduction='none')*torch.from_numpy(weight)).sum()+.5*base.solver.REG*m.weights.square().sum();loss.backward()
    le=abs(float(loss)-value);ge=float(abs(m.weights.grad.numpy()-gradient).max());assert le<=1e-9 and ge<=1e-9
    fd=[]
    for j in np.linspace(0,len(w)-1,8,dtype=int):
        a=w.copy();c=w.copy();a[j]+=1e-5;c[j]-=1e-5
        fd.append(abs((base.solver.objective(a,phi,y,weight)[0]-base.solver.objective(c,phi,y,weight)[0])/2e-5-gradient[j]))
    assert max(fd)<=1e-7;tic=time.monotonic()
    for _ in range(20):base.solver.objective(w,phi,y,weight)
    cost=20+2*800*(time.monotonic()-tic)/20
    contract.put(OUT,'engineering.json',dict(passed=cost<=600,cost_bound_seconds=cost,loss_error=le,gradient_error=ge,finite_difference_error=max(fd)))
    assert cost<=600;contract.verify(OUT);tic=time.monotonic();w,report=base.solver.solve(phi,y,weight);m=model()
    with torch.no_grad():m.weights.copy_(torch.from_numpy(w));q=m(torch.from_numpy(x)).numpy()
    score_error=float(abs(q-phi@w).max());assert score_error<=1e-9
    old=torch.load(oldcp,map_location='cpu',weights_only=False)['state_dict']
    for key in ['mu','sd','delta_scale']:assert torch.equal(old[key],m.state_dict()[key])
    old_objective,_=base.solver.objective(old['weights'].numpy(),phi,y,weight)
    assert report['objective']<=old_objective+1e-8
    report.update(score_error=score_error,old_F_on_augmented_objective=old_objective,numerical_gate_passed=report['projected_gradient_max']<=1e-5)
    assert time.monotonic()-tic<600
    torch.save(dict(state_dict=m.state_dict(),dataset='imdb',base_repeat=REPEAT,augmentation_repeat=REPEAT+1,augmentation='cycle_inverse_256',normalization_changed=False),OUT/'checkpoint.pt')
    np.savez_compressed(OUT/'training_scores.npz',logits=q,targets=y,weights=weight)
    contract.put(OUT,'report.json',report);contract.put(OUT,'completion.json',dict(completed=True,new_coefficient_fits=1,new_original_MLP_fits=0,seconds=time.monotonic()-tic))
    tic=time.monotonic();xx,yy,ww,pp=data();replay,rr=base.solver.solve(pp,yy,ww)
    err=float(abs(replay-w).max());assert err<=1e-10 and abs(rr['objective']-report['objective'])<=1e-10 and rr['function_gradient_calls']==report['function_gradient_calls']
    assert time.monotonic()-tic<600;contract.verify(OUT)
    contract.put(OUT,'summary.json',dict(passed=report['numerical_gate_passed'],coefficient_fits=1,normalization_changed=False,Q_changed=False,new_test_starts_required=True,base_bank_repeat=REPEAT,augmentation_repeat=REPEAT+1))
    contract.put(OUT,'acceptance.json',dict(passed=True,coefficient_replay_error=err,outputs={str(OUT/'checkpoint.pt'):b.sha(OUT/'checkpoint.pt')},seconds=time.monotonic()-tic))
    print('Cycle coverage fit accepted',report['projected_gradient_max'],flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument("repeat",type=int,choices=[1,2,3]);args=parser.parse_args()
    try:main(args.repeat)
    except BaseException as exc:
        if OUT is not None and OUT.exists():contract.put(OUT,'failure.json',dict(type=type(exc).__name__,message=str(exc)))
        raise
