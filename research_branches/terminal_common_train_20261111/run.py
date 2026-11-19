"""Fixed 20-update CPU finite terminal-objective comparison. No production model."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import hashlib,importlib.util,json,resource,signal,time,traceback
from pathlib import Path
import numpy as np
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/terminal_common_train_20261111'
def load(n,p):
    spec=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
a=load('align',ROOT/'research_branches/toy_gradient_alignment_20261111/run.py')
s=load('common',ROOT/'research_branches/terminal_common_direction_20261111/run.py')
r=a.r
def terminal(f,C,w):
    dist=f.prior.copy();der=np.zeros((64,len(w)))
    for t in range(100,0,-1):
        F=f.features[t];p=expit(F@w);q=f.k[t][0]+p*(f.k[t][1]-f.k[t][0])
        P=np.prod(np.where(f.B[None,:,:]>0,q[:,None,:],1-q[:,None,:]),axis=2)
        dq=(f.k[t][1]-f.k[t][0])*p*(1-p)
        DP=P[:,:,None]*np.einsum('ije,ied->ijd',(f.B[None,:,:]-q[:,None,:])/(q*(1-q))[:,None,:],dq[:,:,None]*F)
        der=P.T@der+np.einsum('i,ijd->jd',dist,DP);dist=dist@P
    return dist@C,C.T@der,dist
def propose(G,arm,target=None):
    b=G.mean(0);bn=np.linalg.norm(b);cert={}
    if arm=='common':
        gn=np.linalg.norm(G,axis=1)
        if min(gn)<=0:raise ArithmeticError('zero risk gradient: stop rather than silently alter common rule')
        cert=s.simplex_min_norm(G/gn[:,None])
        if cert['status']!='strict_descent':raise ArithmeticError('strict common direction not certified')
        delta=.2*bn*np.array(cert['direction'])
        assert max(G@delta)<0
    elif arm=='mean':delta=-.2*b
    else:
        if bn==0:raise ArithmeticError('zero mean cannot norm match')
        delta=-target*b/bn
    return delta,cert
def main():
    start=time.time();OUT.mkdir(exist_ok=False);r.OUT=OUT
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('600 second bound')));signal.alarm(600)
    resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3))
    receipt=dict(status='running',toy_updates=0,GPU_hours=0,production_calls=0,new_sampled_paths=0);hist={};updates=[];finals=[];checks=[];distbank={}
    try:
        f,C,tri,cfg,_=r.setup();f=a.m.augment(f)
        source=ROOT/'results/toy_typed_endpoint_20261110/histories.npz';old=np.load(source)
        paths=[Path(__file__),Path(__file__).with_name('watch.py'),Path(__file__).with_name('audit.py'),ROOT/'research_branches/toy_gradient_alignment_20261111/run.py',ROOT/'research_branches/terminal_common_direction_20261111/run.py',ROOT/'research_branches/toy_typed_endpoint_20261110/run.py',ROOT/'research_branches/full100_toy_20261110/run.py',source,ROOT/'results/terminal_common_direction_20261111/conditions.json']
        for p in paths:r.bind(p)
        protocol=dict(track='CORE finite exact-terminal objective experiment',seeds=r.SEEDS,arms=['common','mean','norm_matched_mean'],updates=20,LR=.2,EMA=.99,representation='fixed corrected 10-parameter typed endpoint features',initialization='same three untrained saved states, new coefficient zero',objective='same five terminal hard risks over full T100 reverse distribution; differs from prior one-step clean training objective',common='row-normalized exact terminal gradients, fixed max-min strict direction, displacement .2*norm(mean original gradients)*unit common direction',norm_control='own mean direction, each step norm copied from paired common arm; experimental control, not deployable rule',readouts=[0,5,10,20],training_randomness='none; exact state integration; no sampled inputs or graphs',no_accept_reject=True,no_CE=True,no_Adam=True,stop='numerical/state failure or uncertified strict common solution; not quality decrease',limits=dict(seconds=600,threads=1,RSS_MiB=2048,GPU=0),bindings=r.BIND)
        r.dump('protocol.json',protocol);ph=hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest();r.dump('freeze.json',dict(epoch=time.time(),protocol_sha256=ph))
        for seed in r.SEEDS:
            w0=old[f'{seed}_typed_endpoint_parameters'][0];norms=[]
            # Pre-update derivative validation against prior independently checked implementation.
            value,G,dist=terminal(f,C,w0);prior=a.full(f,C,w0)
            err=float(abs(G-prior['Gterm']).max());assert err<1e-11
            d,_=propose(G,'common');unit=d/np.linalg.norm(d);h=1e-5
            fd=(a.values(f,C,w0+h*unit)[0]-a.values(f,C,w0-h*unit)[0])/(2*h)
            fe=float(abs(fd-G@unit).max());assert fe<1e-7
            checks.append(dict(seed=seed,kind='initial_gradient_gate',gradient_error=err,finite_difference_error=fe))
            for arm in protocol['arms']:
                w=w0.copy();ema=w.copy();wh=[];eh=[]
                for n in range(21):
                    wh.append(w.copy());eh.append(ema.copy());val,G,dist=terminal(f,C,w)
                    if n in protocol['readouts']:
                        for rd,ww in [('raw',w),('EMA',ema)]:
                            dd=dist if rd=='raw' else terminal(f,C,ww)[2]
                            finals.append(dict(seed=seed,arm=arm,update=n,readout=rd,**r.metrics(f,C,tri,dd)));distbank[f'{seed}_{arm}_{n}_{rd}']=dd
                    if n==20:break
                    delta,cert=propose(G,arm,norms[n] if arm=='norm_matched_mean' else None)
                    if arm=='common':norms.append(float(np.linalg.norm(delta)))
                    if arm=='norm_matched_mean':assert abs(np.linalg.norm(delta)-norms[n])<1e-14
                    nw=w+delta;ne=.99*ema+.01*nw
                    if n==10:
                        ck=OUT/f'{seed}_{arm}_u10.npz';np.savez(ck,theta=w,EMA=ema,cursor=n)
                        with np.load(ck) as z:
                            assert int(z['cursor'])==n;_,GG,_=terminal(f,C,z['theta']);dx,_=propose(GG,arm,norms[n] if arm=='norm_matched_mean' else None)
                            np.testing.assert_array_equal(z['theta']+dx,nw);np.testing.assert_array_equal(.99*z['EMA']+.01*(z['theta']+dx),ne)
                        checks.append(dict(seed=seed,arm=arm,kind='recovery',passed=True,stochastic_streams='none consumed'))
                    after=a.values(f,C,nw)[0];assert np.isfinite(nw).all() and np.isfinite(after).all()
                    updates.append(dict(seed=seed,arm=arm,update=n+1,before=val.tolist(),after=after.tolist(),gradient=G.tolist(),delta=delta.tolist(),norm=float(np.linalg.norm(delta)),first_order=(G@delta).tolist(),actual_change=(after-val).tolist(),certificate=cert))
                    w,ema=nw,ne;receipt['toy_updates']+=1
                hist[f'{seed}_{arm}_parameters']=np.array(wh);hist[f'{seed}_{arm}_EMA']=np.array(eh)
                r.dump('progress.json',dict(seed=seed,arm=arm,updates=receipt['toy_updates'],seconds=time.time()-start))
        for p,h in r.BIND.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
        assert hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest()==ph
        receipt.update(status='complete',bindings_unchanged=True,readout_distributions=len(finals))
    except BaseException as e:receipt.update(status='failed',error=repr(e),traceback=traceback.format_exc())
    finally:
        r.dump('updates.json',updates);r.dump('finals.json',finals);r.dump('checks.json',checks);np.savez_compressed(OUT/'histories.npz',**hist);np.savez_compressed(OUT/'distributions.npz',**distbank)
        receipt.update(seconds=time.time()-start,RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024);r.dump('receipt.json',receipt);print(json.dumps(receipt))
    return 0 if receipt['status']=='complete' else 1
if __name__=='__main__':raise SystemExit(main())
