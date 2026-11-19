"""Exact 64-state CPU learning trial. Formal O/A code and assets stay untouched."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import csv, hashlib, importlib.util, json, resource, signal, time, traceback
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from scipy.special import logit

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'results/one_step_to_terminal_cpu_20261110'
START=time.monotonic()
BIND={};CHECKS=[];COUNT=0

def bind(p):
    p=Path(p);BIND[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return p

def module(name,p):
    spec=importlib.util.spec_from_file_location(name,bind(p));m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m);return m

def save(name,x):
    (OUT/(name+'.json')).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    if isinstance(x,list) and x:
        with (OUT/(name+'.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,list(dict.fromkeys(k for r in x for k in r)))
            w.writeheader();w.writerows(x)

def check(name,error,tol=1e-10):
    CHECKS.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(error<=tol)))
    assert error<=tol,(name,error)

def local(f,theta,details=False):
    _,_,clean,pred=f.matrices(theta)
    risk=0.;g=np.zeros(f.D);blocks=np.zeros(len(f.block_names));rows=[]
    for t in range(1,f.T+1):
        cp=np.exp(clean[t]);joint=f.qref[t][:,None]*cp
        val=joint@f.cost
        # derivative of log p(B|Xt), no derivative through fixed forward qref.
        edge_g=np.einsum('ij,j,ije->ie',joint,f.cost,f.B[None,:,:]-pred[t][:,None,:])
        g+=np.einsum('ie,ied->d',edge_g,f.features[t])/f.T
        risk+=val.sum()/f.T
        block=f.qref[t]@cp@f.blocks;blocks+=block/f.T
        if details:rows.append(dict(t=t,forward_clean_risk=float(val.sum()),
                                   **dict(zip(f.block_names,block.tolist()))))
    return float(risk),g,blocks,rows

def full_eval(f,theta):
    ev=f.evaluate(theta)
    risk,g,blocks,rows=local(f,theta,True)
    P,_,clean,_=f.matrices(theta)
    occ={f.T:f.prior}
    for t in range(f.T,0,-1):occ[t-1]=occ[t]@P[t]
    for row in rows:
        t=row['t'];cp=np.exp(clean[t])
        row.update(generated_state_clean_risk=float(occ[t]@cp@f.cost),
                   state_TV_to_forward_reference=float(np.abs(occ[t]-f.qref[t]).sum()/2))
    d=ev['distribution'];orbit=np.bincount(f.orbits,weights=d)
    metrics=dict(ev['metrics'])
    # Old evaluate.single_step_clean_risk uses generated-state occupancy: rename explicitly.
    metrics['generated_state_clean_risk']=metrics.pop('single_step_clean_risk')
    metrics.update(forward_clean_risk=risk,reference_graph_probability=float(d[f.refid]),
                   max_graph_probability=float(d.max()),
                   isomorph_entropy=float(-np.sum(orbit[orbit>0]*np.log(orbit[orbit>0]))))
    metrics.update({'forward_'+k:float(v) for k,v in zip(f.block_names,blocks)})
    return ev,metrics,rows,g

def main():
    global COUNT
    OUT.mkdir(exist_ok=False)
    cfg=json.loads(bind(HERE/'protocol.json').read_text())
    signal.alarm(cfg['limits']['worker_seconds'])
    resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3))
    bind(Path(__file__));save('protocol',cfg)
    save('start',dict(utc=datetime.now(timezone.utc).isoformat(),GPU=0,threads=1))
    m=module('finite_structure_transfer',ROOT/'research_branches/terminal_structure_audit_20261109/audit_cpu.py')
    m.kernel_module=module('bound_reverse_transfer',m.KERNEL)
    old=json.loads(bind(m.PROTOCOL).read_text());f=m.FiniteDiffusion(old)
    assert cfg['initializations']==old['initializations']
    assert cfg['learning_rate']==old['trial']['learning_rate']
    assert f.S==64 and f.T==3
    check('t1_K0',np.max(abs(f.k[1][0])))
    check('t1_K1',np.max(abs(f.k[1][1]-1)))
    for t in range(1,f.T+1):check('forward_mass_'+str(t),abs(f.qref[t].sum()-1))
    initials=[]
    for seed in cfg['initializations']:
        theta=np.random.default_rng(seed).normal(0,.15,f.D);theta[:3]+=logit(f.pi)
        initials.append(theta)
        ev=f.evaluate(theta);risk,g,_,_=local(f,theta)
        for t,p in ev['P'].items():check(f'kernel_mass_{seed}_{t}',np.max(abs(p.sum(1)-1)))
        eps=1e-5;fd=[];fdce=[]
        for k in range(f.D):
            delta=np.eye(f.D)[k]*eps
            fd.append((local(f,theta+delta)[0]-local(f,theta-delta)[0])/(2*eps))
            fdce.append((f.evaluate(theta+delta)['metrics']['reference_edge_CE']-
                         f.evaluate(theta-delta)['metrics']['reference_edge_CE'])/(2*eps))
        check(f'local_gradient_{seed}',np.max(abs(g-np.array(fd))),1e-7)
        check(f'CE_gradient_{seed}',np.max(abs(ev['ce']-np.array(fdce))),1e-7)
    save('checks',CHECKS)
    save('gate',dict(passed=True,checks=len(CHECKS),scope='exact finite gradients/identities, not quality'))
    curves=[];stages=[];updates=[];saved=[];dist=[];thetas=[];chains=[]
    for seed,initial in zip(cfg['initializations'],initials):
        for arm in cfg['arms']:
            theta=initial.copy()
            for step in range(cfg['updates']+1):
                ev,metrics,per_t,g=full_eval(f,theta)
                assert np.isfinite(theta).all() and np.isfinite(g).all()
                if step in cfg['evaluation_steps']:
                    row=dict(seed=seed,arm=arm,step=step,**metrics,
                             displacement_from_initial=float(np.linalg.norm(theta-initial)))
                    curves.append(row)
                    stages.extend([dict(seed=seed,arm=arm,step=step,**r) for r in per_t])
                    saved.append(dict(seed=seed,arm=arm,step=step));dist.append(ev['distribution']);thetas.append(theta.copy())
                    chains.append(np.stack([ev['P'][t] for t in range(1,f.T+1)]))
                if step==cfg['updates']:break
                grad=ev['ce'] if arm=='reference_edge_CE' else g
                displacement=-cfg['learning_rate']*grad
                updates.append(dict(seed=seed,arm=arm,step=step+1,gradient_norm=float(np.linalg.norm(grad)),
                    displacement_norm=float(np.linalg.norm(displacement)),
                    local_risk_directional_derivative=float(g@displacement),
                    terminal_risk_directional_derivative=float(ev['g']@displacement)))
                theta=theta+displacement;COUNT+=1
            save('curves',curves);save('stages',stages);save('updates',updates)
    assert COUNT==cfg['total_arithmetic_updates']
    finals=[]
    for seed in cfg['initializations']:
        for arm in cfg['arms']:
            a=next(r for r in curves if r['seed']==seed and r['arm']==arm and r['step']==0)
            b=next(r for r in curves if r['seed']==seed and r['arm']==arm and r['step']==cfg['updates'])
            finals.append(dict(seed=seed,arm=arm,**{k:v for k,v in b.items() if k not in ['seed','arm','step']},
                  **{'change_'+k:b[k]-a[k] for k in b if k not in ['seed','arm','step']}))
    save('finals',finals);save('distribution_index',saved)
    np.savez_compressed(OUT/'states_and_distributions.npz',bits=f.B,cost=f.cost,blocks=f.blocks,
        orbits=f.orbits,triangles=f.tri,components=f.components,reference_id=f.refid,
        prior=f.prior,distributions=np.array(dist),parameters=np.array(thetas),
        transitions=np.array(chains),qref=np.stack([f.qref[t] for t in range(1,f.T+1)]))
    for p,h in BIND.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    save('bindings',BIND)
    save('receipt',dict(status='complete_stopped',seconds=time.monotonic()-START,
        RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,CPU_threads=1,GPU_hours=0,
        formal_model_calls=0,formal_training_updates=0,toy_arithmetic_parameter_updates=COUNT,
        random_graphs=0,real_generated_graphs=0,exact_graph_support=f.S,diffusion_steps=f.T,
        checks=len(CHECKS),all_checks_passed=True,original_assets_unchanged=True))
    print(json.dumps(dict(receipt=json.loads((OUT/'receipt.json').read_text()),finals=finals)))

if __name__=='__main__':
    try:main()
    except BaseException:
        if OUT.exists():save('failure',dict(traceback=traceback.format_exc(),seconds=time.monotonic()-START,
                                          toy_updates_completed=COUNT))
        raise
