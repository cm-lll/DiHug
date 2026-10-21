"""Frozen-weight generation comparison; unique module name avoids old common.py."""
import os,sys,json,time,hashlib
from pathlib import Path
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
HERE=Path(__file__).resolve().parent;EXP=HERE.parent
sys.path.insert(0,str(EXP/'role_logscale_reader_v1'))
import diagnose as old
sys.path.insert(0,str(EXP/'role_newton_reader_v1'))
import solver as newton
c=old.c;np=c.np;torch=c.torch
OUT=Path('/data2/lyh/gnn_project/DiHug/results/role_newton_generation_20261021_v1')
read=c.read;sha=c.sha
def save(p,x):
    p=Path(p);q=p.with_name(p.name+'.tmp')
    q.write_text(json.dumps(x,indent=2,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else (_ for _ in ()).throw(TypeError(type(v).__name__)))+'\n');q.replace(p)
SEEDS=[2026102011,2026102012,2026102013];ARMS=['DP','Newton']
def check():
    if (OUT/'STOP').exists():raise InterruptedError('STOP')
    if time.time()>=read(OUT/'activation.json')['deadline_epoch']-120:raise TimeoutError('unique45min reserve120')
def sources():
    x=dict(read(c.RESULT/'freeze.json')['source'])
    for folder in (HERE,EXP/'role_logscale_reader_v1',EXP/'role_newton_reader_v1'):
        for p in folder.glob('*'):
            if p.is_file() and p.suffix in ('.py','.json','.md'):x[str(p.relative_to(EXP))]=sha(p)
    return x
def frozen():
    f=read(OUT/'freeze.json')
    for n,h in f['source'].items():assert sha(EXP/n)==h,n
    for n,h in f['inputs'].items():assert sha(c.RESULT/n)==h,n
    assert f['activation_sha256']==sha(OUT/'activation.json')
    return f
def verify(P,LP,L,counts,r):
    assert r.get('new_contract_pass') and np.isfinite(P).all() and (P>=0).all()
    row=float(abs(P.sum(1)-1).max());col=float((abs(P.sum(0)-counts)/np.maximum(1,counts)).max())
    assert row<=1e-8 and col<=1e-8
    finite=np.isfinite(LP);rounded=finite&(P==0)
    assert np.all(LP[rounded]<-700) and np.all(P[~finite]==0)
    np.testing.assert_array_equal(P[finite&~rounded],np.exp(LP[finite&~rounded]))
    mass=rounded.sum(1)*np.exp(-700.)
    assert mass.max(initial=0)<=1e-280 and mass.sum()<=1e-276
    assert int(rounded.sum())==r['rounded_positive_count']
    from scipy.special import logsumexp
    assert abs(logsumexp(LP,axis=1)).max()<=2e-12
    kkt=None
    if finite.all():
        D=LP-L;kkt=float(abs(D-D[:,:1]-D[:1,:]+D[0,0]).max());assert kkt<=2e-10
    return dict(independent_row=row,independent_column=col,independent_KKT=kkt,
                independent_tail_mass=float(mass.sum()),tail_row_mass=mass.tolist())

class Reader:
    def __init__(self,arm,folder):self.arm=arm;self.folder=folder;self.rows=[];self.arrays={};self.step=0;self.typ=0;self.original=old.cap.allocation_marginals
    def begin(self,t):self.step=t;self.typ=0;self.current=[]
    def __call__(self,L,counts):
        check();start=time.perf_counter();typ=self.typ;self.typ+=1
        if self.arm=='Newton':
            P,LP,r=newton.solve(L,counts,check)
            if not r.get('new_contract_pass'):
                np.savez_compressed(self.folder/'failed_read.npz',L=L,counts=counts,t=self.step,type=typ)
                save(self.folder/'failed_read.json',r);raise newton.ReadoutFailure(str(r))
            independent=verify(P,LP,L,counts,r)
            for key,v in [('L',L),('P',P),('logP',LP)]:self.arrays.setdefault(f'type{typ}_{key}',[]).append(v.copy())
        else:
            P,r=self.original(L,counts)
            assert np.max(abs(P.sum(1)-1))<=1e-8 and np.max(abs(P.sum(0)-counts)/np.maximum(1,counts))<=1e-8
            independent={};r=dict(r,status='exact_DP',row_residual=float(abs(P.sum(1)-1).max()),column_residual=float((abs(P.sum(0)-counts)/np.maximum(1,counts)).max()))
        rr=dict(t=self.step,type=typ,n=len(L),k=L.shape[1],**r,**independent,total_checked_seconds=time.perf_counter()-start)
        self.current.append(rr);self.rows.append(rr);return P,r
    def install(self,m):
        old.cap.allocation_marginals=self
        def hook(module,args,value):
            assert len(self.current)==4
            for r,v in zip(self.current,self.values):
                assert np.isfinite(v).all()
                mass=np.asarray(r.get('tail_row_mass',[0.]*r['n']))
                r['role_value_max']=float(abs(v).max());r['context_tail_bound']=float((mass*abs(v).max()).max(initial=0))
                assert r['context_tail_bound']<=1e-250
        def vh(module,args,value):self.values.append(value.detach().cpu().numpy().copy())
        self.values=[];self.hooks=[m.branch.value.register_forward_hook(vh),m.branch.register_forward_hook(hook)]
    def close(self):
        old.cap.allocation_marginals=self.original
        for h in self.hooks:h.remove()
