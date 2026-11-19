"""Matched original-CE replay with uniform versus bidirectional corruption coverage."""
from cache import *
import copy
import hashlib
import torch.nn.functional as F

EPOCH=ROOT/'results/typed_edit_epoch_gate_20261117'
SEEDS=[810,811]
STEPS=200


def checkpoint(seed):
    p=EPOCH/f'fits/seed{seed}/M';ep=b.read(p/'selection.json')['epoch']
    return p/f'epoch{ep}.pt'


def model(seed):
    m=b.model(45,'M',seed,'cpu')
    m.load_state_dict(torch.load(checkpoint(seed),map_location='cpu',weights_only=False)['state_dict'])
    return m


def optimizer(m):return torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=1e-4)
def seed_rng(seed):torch.manual_seed(920000+seed)
def rng_hash():return hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()


def data():
    z=np.load(b.RUN/'cache/pubmed/data.npz');old={k:z[k][z['ids']<30] for k in ['x','y','ids']}
    new={kind:{k:v for k,v in np.load(OUT/f'train_{kind}.npz').items()} for kind in ['U','B']}
    v=np.load(EPOCH/'validation.npz');validation={k:v[k][v['ids']>=16] for k in ['x','y','ids']}
    fresh={k:v for k,v in np.load(OUT/'validation.npz').items()}
    return old,new,{'old_assessment':validation,'new_assessment':fresh}


def batches(seed):
    rng=np.random.default_rng(920000+seed)
    return [(rng.choice(3000,32,replace=False),rng.choice(256,32,replace=False)) for _ in range(STEPS)]


def batch(old,new,indices):
    i,j=indices
    return torch.from_numpy(np.concatenate([old['x'][i],new['x'][j]])),torch.from_numpy(np.concatenate([old['y'][i],new['y'][j]]))


def update(m,o,x,y):
    m.train();o.zero_grad(set_to_none=True)
    loss=F.cross_entropy(m(x),y);assert torch.isfinite(loss)
    loss.backward();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),5.,error_if_nonfinite=True);o.step()
    assert all(torch.isfinite(p).all() for p in m.parameters())
    return float(loss.detach()),float(norm)


def snapshot(m,o):return copy.deepcopy(dict(state_dict=m.state_dict(),optimizer=o.state_dict(),rng=torch.get_rng_state()))
def equal(a,z):
    if isinstance(a,torch.Tensor):return torch.equal(a,z)
    if isinstance(a,dict):return a.keys()==z.keys() and all(equal(a[k],z[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(z) and all(equal(x,y) for x,y in zip(a,z))
    return a==z


def verify_train():
    verify()
    for p,h in b.read(OUT/'training_manifest.json')['files'].items():assert b.sha(p)==h,p


def predict(m,splits,folder):
    folder.mkdir(parents=True,exist_ok=True)
    for name,z in splits.items():
        _,q,_=b.c.eval_model(m,z['x'],z['y'],'cpu')
        np.savez_compressed(folder/f'{name}.npz',logits=q,targets=z['y'],ids=z['ids'])


def gate(old,new,splits):
    start=time.monotonic();error=0.
    for seed in SEEDS:
        m=model(seed);z=splits['old_assessment'];_,q,_=b.c.eval_model(m,z['x'],z['y'],'cpu')
        ep=b.read(EPOCH/f'fits/seed{seed}/M/selection.json')['epoch']
        saved=np.load(EPOCH/f'fits/seed{seed}/M/assessment_e{ep}.npz')['logits']
        error=max(error,float(np.max(abs(q-saved))))
    assert error<=2e-5
    runs=[]
    for _ in range(2):
        m=model(810);o=optimizer(m);seed_rng(810)
        for ix in batches(810)[:8]:update(m,o,*batch(old,new['B'],ix))
        runs.append(snapshot(m,o))
    assert equal(*runs)
    torch.save(runs[1],OUT/'training_engineering_resume.pt')
    x,y=batch(old,new['B'],batches(810)[8]);update(m,o,x,y);expected=snapshot(m,o)
    saved=torch.load(OUT/'training_engineering_resume.pt',map_location='cpu',weights_only=False)
    m.load_state_dict(saved['state_dict']);o.load_state_dict(saved['optimizer']);torch.set_rng_state(saved['rng'])
    update(m,o,x,y);assert equal(expected,snapshot(m,o))
    times=[]
    for kind in ['U','B']:
        m=model(810);o=optimizer(m);seed_rng(810);tic=time.monotonic()
        for ix in batches(810)[:8]:update(m,o,*batch(old,new[kind],ix))
        times.append((time.monotonic()-tic)/8)
    tic=time.monotonic()
    for z in splits.values():b.c.eval_model(m,z['x'],z['y'],'cpu')
    eval_time=time.monotonic()-tic
    bound=2*(800*max(times)+6*eval_time)+30
    put('training_gate.json',dict(passed=bound<=240,conservative_seconds=bound,original_logit_max_error=error,
        repeat_eight_updates_exact=True,serialized_next_step_exact=True,engineering_updates=34,
        update_seconds=times,eval_seconds=eval_time,wall_seconds=time.monotonic()-start))
    assert bound<=240


def main():
    b.setup();assert b.read(OUT/'cache_acceptance.json')['passed']
    assert not (OUT/'training_manifest.json').exists()
    old,new,splits=data()
    files=[*Path(__file__).parent.glob('*.py'),Path(__file__).with_name('PROTOCOL.md'),
        OUT/'cache_acceptance.json',OUT/'train_U.npz',OUT/'train_B.npz',OUT/'validation.npz',
        b.RUN/'cache/pubmed/data.npz',EPOCH/'validation.npz']
    for seed in SEEDS:
        ep=b.read(EPOCH/f'fits/seed{seed}/M/selection.json')['epoch']
        files += [checkpoint(seed),EPOCH/f'fits/seed{seed}/M/selection.json',EPOCH/f'fits/seed{seed}/M/assessment_e{ep}.npz']
    put('training_manifest.json',dict(files={str(p):b.sha(p) for p in files},seeds=SEEDS,updates=200,
        ordinary_ce=True,learning_rate=1e-4,old_rows_per_batch=32,new_rows_per_batch=32))
    gate(old,new,splits);verify_train();start=time.monotonic();reports=[]
    for seed in SEEDS:
        predict(model(seed),splits,OUT/f'fits/{seed}/base')
        order=batches(seed);order_hash=hashlib.sha256(b''.join(i.tobytes()+j.tobytes() for i,j in order)).hexdigest()
        pair=[]
        for kind in ['U','B']:
            m=model(seed);o=optimizer(m);seed_rng(seed);before=rng_hash();history=[];tic=time.monotonic()
            for step,ix in enumerate(order,1):
                assert time.monotonic()-start<240
                loss,norm=update(m,o,*batch(old,new[kind],ix));history.append(dict(step=step,loss=loss,grad_norm=norm))
            after=rng_hash();dest=OUT/f'fits/{seed}/{kind}';dest.mkdir(parents=True,exist_ok=False)
            torch.save({**snapshot(m,o),'updates':200},dest/'checkpoint.pt');predict(m,splits,dest)
            report=dict(seed=seed,kind=kind,updates=200,initial_checkpoint_sha256=b.sha(checkpoint(seed)),
                checkpoint_sha256=b.sha(dest/'checkpoint.pt'),batch_order_sha256=order_hash,
                dropout_rng_start=before,dropout_rng_end=after,learning_rate=1e-4,
                wall_seconds=time.monotonic()-tic,history=history)
            put(Path('fits')/str(seed)/kind/'report.json',report);pair.append(report)
            reports.append({k:v for k,v in report.items() if k!='history'})
            print(seed,kind,round(time.monotonic()-start,2),flush=True)
        for k in ['initial_checkpoint_sha256','batch_order_sha256','dropout_rng_start','dropout_rng_end']:
            assert pair[0][k]==pair[1][k],k
    verify_train()
    put('training_completion.json',dict(completed=True,scientific_updates=800,engineering_updates=34,
        runs=reports,seconds=time.monotonic()-start,new_independent_fitting_seeds=0,new_generated_graphs=0,gpu_calls=0))


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        put('training_failure.json',dict(type=type(exc).__name__,message=str(exc)));raise
