"""Bounded finite toy input-information experiment; no production model calls."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import copy,hashlib,importlib.util,itertools,json,resource,signal,time,traceback
from pathlib import Path
import numpy as np
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/toy_typed_endpoint_20261110'
def load(p):
    s=importlib.util.spec_from_file_location('base',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
r=load(ROOT/'research_branches/full100_toy_20261110/run.py');r.OUT=OUT
OLD=ROOT/'results/mean_anchored_toy_20261110_attempt2'
def augment(f):
    g=copy.deepcopy(f);d=f.adj.sum(-1);z=np.zeros((64,6))
    for e,(i,j) in enumerate(f.edges):
        if f.types[i]!=f.types[j]:
            a,b=(i,j) if f.types[i]<f.types[j] else (j,i)
            z[:,e]=(d[:,a]-d[:,b])/3
    g.features={t:np.concatenate([F,z[:,:,None]],axis=2) for t,F in f.features.items()};g.D=10
    return g
def terminal(f,C,tri,w):
    dist=f.prior.copy()
    for t in range(100,0,-1):dist=dist@r.transitions(f,w,t)[0]
    return r.metrics(f,C,tri,dist),dist
def feature_gate(f,g):
    # Independently check typed relabeling equivariance and elimination of accidental ties.
    checks=0
    for s,A in enumerate(f.adj):
        for perm in ((0,1,2,3),(1,0,2,3),(0,1,3,2),(1,0,3,2)):
            B=A[np.ix_(perm,perm)];bits=B[f.edges[:,0],f.edges[:,1]]
            ss=np.flatnonzero((f.B==bits).all(1))[0]
            for e,(i,j) in enumerate(f.edges):
                a,b=sorted((perm[i],perm[j]));ee=np.flatnonzero((f.edges==[a,b]).all(1))[0]
                np.testing.assert_array_equal(g.features[1][ss,e],g.features[1][s,ee]);checks+=1
    assert f.features[1][2,2].tolist()==f.features[1][2,3].tolist()
    assert g.features[1][2,2,-1]!=g.features[1][2,3,-1]
    return checks
def main():
    start=time.time();OUT.mkdir(exist_ok=False);r.OUT=OUT
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('600 second limit')));signal.alarm(600)
    receipt=dict(status='running',GPU_hours=0,production_calls=0,toy_updates=0,new_sampled_paths=0)
    finals=[];updates=[];hist={};validation=[];dists={};recovery=[]
    try:
        f,C,tri,cfg,_=r.setup();g=augment(f)
        paths=[Path(__file__),Path(__file__).with_name('watch.py'),Path(__file__).with_name('audit.py'),ROOT/'research_branches/full100_toy_20261110/run.py',OLD/'training_trajectories.npz',OLD/'protocol.json',ROOT/'results/full100_direction_long_20261110/input_banks.npz',ROOT/'results/full100_adam_scale_20261110/calibration_and_eval.npz']
        for p in paths:r.bind(p)
        bank=np.load(paths[-2]);ev=np.load(paths[-1]);old=np.load(paths[4])
        protocol=dict(track='CORE finite toy diagnostic',arms=['original','typed_endpoint'],seeds=r.SEEDS,steps=100,LR=.2,optimizer='plain SGD of mean of unchanged five exact expected hard costs',K=8,EMA=.99,T=100,feature='append signed total-degree difference by immutable endpoint type, /3; same-type edges zero',initialization='original 9 parameters retained; new coefficient zero; identical initial predictions',capacity='9 versus 10 active parameters; information and capacity change together',readouts=[20,50,100],validation='existing separate 96 inputs; development only',limits=dict(CPU_threads=1,seconds=600,memory_MiB=2048,GPU=0),objective_names=['relation_count','joint_degree','triangle001','triangle011','component_cost'],bindings=r.BIND)
        r.dump('protocol.json',protocol);ph=hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest();r.dump('freeze.json',dict(epoch=time.time(),protocol_sha256=ph))
        equiv=feature_gate(f,g);fd=[];initial_errors=[]
        for seed in r.SEEDS:
            w0=old[f'{seed}_average_parameters'][0];v0=np.r_[w0,0.]
            initial_errors.append(float(max(np.max(abs(f.features[t]@w0-g.features[t]@v0)) for t in range(1,101))))
            assert initial_errors[-1]<1e-14
            ts,xs=bank[f'{seed}_t'],bank[f'{seed}_x'];et,ex=ev[f'{seed}_eval_t'],ev[f'{seed}_eval_x']
            for arm,model,init in [('original',f,w0),('typed_endpoint',g,v0)]:
                w=init.copy();ema=w.copy();wh=[];eh=[]
                for n in range(101):
                    wh.append(w.copy());eh.append(ema.copy())
                    if arm=='original':np.testing.assert_allclose(w,old[f'{seed}_average_parameters'][n],atol=1e-14,rtol=1e-14)
                    cost,_,ce,_=r.atomic(model,C,w,et,ex)
                    validation.append(dict(seed=seed,arm=arm,update=n,CE=ce,mean_five_cost=float(cost.mean()),**dict(zip(protocol['objective_names'],map(float,cost)))))
                    if n in (0,20,50,100):
                        for readout,ww in [('raw',w),('EMA',ema)]:
                            met,dist=terminal(model,C,tri,ww);key=f'{seed}_{arm}_{n}_{readout}'
                            finals.append(dict(seed=seed,arm=arm,update=n,readout=readout,**met));dists[key]=dist
                    if n==100:break
                    co,G,ce,_=r.atomic(model,C,w,ts[n],xs[n]);nw=w-.2*G.mean(0);ne=.99*ema+.01*nw
                    if n in (0,99):
                        for j in range(len(w)):
                            h=np.eye(len(w))[j]*1e-5
                            val=(r.atomic(model,C,w+h,ts[n],xs[n])[0]-r.atomic(model,C,w-h,ts[n],xs[n])[0])/2e-5
                            fd.append(float(np.max(abs(val-G[:,j]))));assert fd[-1]<1e-8
                    if n==50:
                        p=OUT/f'{seed}_{arm}_u50.npz';np.savez(p,theta=w,EMA=ema,cursor=n)
                        with np.load(p) as a:
                            assert int(a['cursor'])==n
                            _,GG,_,_=r.atomic(model,C,a['theta'],ts[n],xs[n]);nn=a['theta']-.2*GG.mean(0)
                            np.testing.assert_array_equal(nn,nw);np.testing.assert_array_equal(.99*a['EMA']+.01*nn,ne)
                        recovery.append(dict(seed=seed,arm=arm,passed=True))
                    assert np.isfinite(nw).all()
                    updates.append(dict(seed=seed,arm=arm,update=n+1,gradient=G.tolist(),norm=float(np.linalg.norm(nw-w)),costs=co.tolist(),CE=ce))
                    w,ema=nw,ne;receipt['toy_updates']+=1
                hist[f'{seed}_{arm}_parameters']=np.array(wh);hist[f'{seed}_{arm}_EMA']=np.array(eh)
        for p,h in r.BIND.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
        assert hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest()==ph
        r.dump('gates.json',dict(passed=True,equivariance_checks=equiv,initial_logit_errors=initial_errors,finite_difference_checks=len(fd),max_fd=max(fd),recovery=recovery,baseline_history_replayed=True))
        receipt.update(status='complete',exact_terminal_distributions=len(finals),bindings_unchanged=True)
    except BaseException as e:receipt.update(status='failed',error=repr(e),traceback=traceback.format_exc())
    finally:
        r.dump('finals.json',finals);r.dump('updates.json',updates);r.table('validation.csv',validation)
        np.savez_compressed(OUT/'histories.npz',**hist);np.savez_compressed(OUT/'distributions.npz',**dists)
        receipt.update(seconds=time.time()-start,RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024);r.dump('receipt.json',receipt);print(json.dumps(receipt))
    return 0 if receipt['status']=='complete' else 1
if __name__=='__main__':raise SystemExit(main())
