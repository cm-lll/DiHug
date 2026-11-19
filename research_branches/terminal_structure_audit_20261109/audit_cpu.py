"""Exact terminal-risk credit assignment on a finite typed diffusion.

No trained O/A model, GPU, random trajectory, external code, or production edits.
The tiny shared arithmetic predictor is optimized only after numerical gates.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import csv, hashlib, importlib.util, itertools, json, resource, signal, time
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from scipy.special import expit, logit
from scipy.optimize import linear_sum_assignment

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'results/terminal_structure_audit_20261109'
OUT=BASE/'cpu'
PROTOCOL=Path(__file__).with_name('protocol.json')
KERNEL=ROOT/'research_branches/cn_multiseed_20261104/research_branches/typed_structure_early_cn_expansion_review/experiments/typed_structure_early_cn_expansion_v1/package/hsgd/kernel.py'
START=time.monotonic()

def save(name,value):
    (OUT/(name+'.json')).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    if isinstance(value,list) and value:
        with (OUT/(name+'.csv')).open('w',newline='') as f:
            writer=csv.DictWriter(f,list(value[0]));writer.writeheader();writer.writerows(value)

class FiniteDiffusion:
    def __init__(self,cfg):
        self.types=np.array(cfg['graph']['node_types'])
        self.edges=np.array(cfg['graph']['unordered_slots'])
        self.ref=np.array(cfg['graph']['reference_bits'],dtype=float)
        self.f=np.array([0 if self.types[i]==self.types[j]==0 else 2 if self.types[i]==self.types[j]==1 else 1 for i,j in self.edges])
        self.pi=np.array(cfg['graph']['family_priors'])
        self.alpha=np.array(cfg['noise']['cumulative_retention']);self.T=len(self.alpha)-1
        self.B=np.array(list(itertools.product([0.,1.],repeat=len(self.edges))))
        self.S,self.M=self.B.shape
        self.adj=np.zeros((self.S,4,4))
        for e,(i,j) in enumerate(self.edges):self.adj[:,i,j]=self.adj[:,j,i]=self.B[:,e]
        self.refid=int(np.where((self.B==self.ref).all(1))[0][0])
        self.prior=np.prod(np.where(self.B>0,self.pi[self.f],1-self.pi[self.f]),axis=1)
        self.kernel=kernel_module.MarginalKernel(self.pi,self.alpha)
        self.features={};self.k={};self.qref={}
        deg=self.adj.sum(-1);cn=self.adj@self.adj
        for t in range(1,self.T+1):
            features=[]
            for e,(i,j) in enumerate(self.edges):
                features.append(np.column_stack([
                    np.repeat(np.eye(3)[self.f[e]][None,:],self.S,axis=0),
                    self.B[:,e],(deg[:,i]+deg[:,j])/6,
                    ((deg[:,i]-deg[:,j])/3)**2,cn[:,i,j]/2,
                    np.full(self.S,t/self.T), self.B.mean(1)]))
            self.features[t]=np.stack(features,axis=1)
            kval=[]
            for clean in (0.,1.):
                p=np.full(self.S*self.M,clean)
                kval.append(self.kernel.reverse(p,self.B.ravel(),np.tile(self.f,self.S),t-1,t).reshape(self.S,self.M))
            self.k[t]=kval
            q=self.alpha[t]*self.ref+(1-self.alpha[t])*self.pi[self.f]
            self.qref[t]=np.prod(np.where(self.B>0,q,1-q),axis=1)
        self.D=self.features[1].shape[-1]
        self.block_names=['relation_count','anonymous_joint_degree','typed_triangles','components']
        self.blocks,self.tri,self.components=self.make_cost()
        self.cost=self.blocks.mean(1)
        self.orbits=self.make_orbits()

    def make_orbits(self):
        permutations=[]
        for a in itertools.permutations([0,1]):
            for b in itertools.permutations([2,3]):permutations.append(np.array(a+b))
        ids=[]
        for A in self.adj:
            keys=[]
            for perm in permutations:
                x=A[np.ix_(perm,perm)]
                keys.append(tuple(x[self.edges[:,0],self.edges[:,1]].astype(int)))
            ids.append(min(keys))
        uniq=sorted(set(ids));return np.array([uniq.index(x) for x in ids])

    def make_cost(self):
        degree=np.zeros((self.S,4,3));counts=np.zeros((self.S,3))
        for e,(i,j) in enumerate(self.edges):
            degree[:,i,self.f[e]]+=self.B[:,e];degree[:,j,self.f[e]]+=self.B[:,e]
            counts[:,self.f[e]]+=self.B[:,e]
        triangles=np.zeros((self.S,2))
        for i,j,k in itertools.combinations(range(4),3):
            typ=int(self.types[[i,j,k]].sum())-1
            triangles[:,typ]+=self.adj[:,i,j]*self.adj[:,i,k]*self.adj[:,j,k]
        comps=[]
        for A in self.adj:
            unseen=set(range(4));n=0
            while unseen:
                todo=[unseen.pop()];n+=1
                while todo:
                    v=todo.pop();nxt={int(i) for i in np.flatnonzero(A[v])}&unseen
                    unseen-=nxt;todo.extend(nxt)
            comps.append(n)
        comps=np.array(comps)
        joint=[]
        for d in degree:
            ds=[]
            for typ in [0,1]:
                ix=np.flatnonzero(self.types==typ)
                C=np.abs(d[ix,None,:]-degree[self.refid,ix][None,:,:]).sum(-1)
                a,b=linear_sum_assignment(C);ds.append(C[a,b].mean()/3)
            joint.append(np.mean(ds))
        blocks=np.column_stack([
            np.mean(((counts-counts[self.refid])/np.maximum(1,counts[self.refid]))**2,axis=1),
            joint,
            np.mean(((triangles-triangles[self.refid])/np.maximum(1,triangles[self.refid]))**2,axis=1),
            ((comps-comps[self.refid])/3)**2])
        return blocks,triangles.sum(1),comps

    def matrices(self,theta):
        P={};DP={};clean={};pred={}
        for t,F in self.features.items():
            p=expit(F@theta);q=self.k[t][0]+p*(self.k[t][1]-self.k[t][0])
            assert (q>0).all() and (q<1).all(), 'nonfinite or boundary predicted probability'
            dq=((self.k[t][1]-self.k[t][0])*p*(1-p))[:,:,None]*F
            lp=np.einsum('je,ie->ij',self.B,np.log(q))+np.einsum('je,ie->ij',1-self.B,np.log1p(-q))
            P[t]=np.exp(lp)
            score=np.einsum('ije,ied->ijd',(self.B[None,:,:]-q[:,None,:])/(q*(1-q))[:,None,:],dq)
            DP[t]=P[t][:,:,None]*score
            clean[t]=np.einsum('je,ie->ij',self.B,np.log(p))+np.einsum('je,ie->ij',1-self.B,np.log1p(-p))
            pred[t]=p
        return P,DP,clean,pred

    def evaluate(self,theta):
        P,DP,clean,pred=self.matrices(theta)
        marg={self.T:self.prior};der=np.zeros((self.S,self.D))
        for t in range(self.T,0,-1):
            der=P[t].T@der+np.einsum('i,ijd->jd',marg[t],DP[t])
            marg[t-1]=marg[t]@P[t]
        terminal=float(marg[0]@self.cost);grad=der.T@self.cost
        downstream={0:np.eye(self.S)}
        for t in range(1,self.T+1):downstream[t]=P[t]@downstream[t-1]
        pg=np.zeros(self.D);constant_pg=np.zeros(self.D)
        eager=np.zeros(self.D);eager_constant=np.zeros(self.D)
        local=np.zeros(self.D);ce=np.zeros(self.D);ce_loss=0.;local_loss=0.
        stages=[]
        for t in range(1,self.T+1):
            value=downstream[t-1]@self.cost
            pg+=np.einsum('i,j,ijd->d',marg[t],value,DP[t])
            constant_pg+=np.einsum('i,ijd->d',marg[t],DP[t])
            # Actual generated endpoint given Xt, not a clean-head independent draw.
            joint=marg[t][:,None]*downstream[t]
            logscore=np.einsum('ije,ied->ijd',self.B[None,:,:]-pred[t][:,None,:],self.features[t])
            et=np.einsum('ij,j,ijd->d',joint,self.cost,logscore)
            ec=np.einsum('ij,ijd->d',joint,logscore)
            eager+=et;eager_constant+=ec
            # Single-step clean risk with frozen on-policy state marginal.
            cp=np.exp(clean[t]);weights=marg[t][:,None]*cp
            local+=np.einsum('ij,j,ijd->d',weights,self.cost,logscore)/self.T
            local_loss+=np.einsum('ij,j->',weights,self.cost)/self.T
            ce_loss-=self.qref[t]@clean[t][:,self.refid]/(self.T*self.M)
            ce+=np.einsum('i,ie,ied->d',self.qref[t],pred[t]-self.ref,self.features[t])/(self.T*self.M)
            stages.append(dict(t=t,occupancy_mass=float(marg[t].sum()),
                               constant_eager_norm=float(np.linalg.norm(ec)),
                               exact_contribution_norm=float(np.linalg.norm(np.einsum('i,j,ijd->d',marg[t],value,DP[t])))))
        orbit=np.bincount(self.orbits,weights=marg[0]);entropy=float(-np.sum(marg[0]*np.log(marg[0])))
        metrics=dict(terminal_cost=terminal,reference_edge_CE=float(ce_loss),
                     single_step_clean_risk=float(local_loss),edges=float(marg[0]@self.B.sum(1)),
                     triangle_mean=float(marg[0]@self.tri),triangle_MAE=float(marg[0]@np.abs(self.tri-self.tri[self.refid])),
                     connected_probability=float(marg[0]@(self.components==1)),
                     labeled_entropy=entropy,effective_labeled_graphs=float(np.exp(entropy)),
                     isomorphic_collision=float(orbit@orbit),reference_isomorph_mass=float(orbit[self.orbits[self.refid]]),
                     **dict(zip(self.block_names,(marg[0]@self.blocks).tolist())))
        return dict(metrics=metrics,g=grad,pg=pg,constant_pg=constant_pg,eager=eager,
                    eager_constant=eager_constant,eager_centered=eager-terminal*eager_constant,
                    local=local,ce=ce,P=P,DP=DP,stages=stages,distribution=marg[0])

def cos(a,b):
    den=np.linalg.norm(a)*np.linalg.norm(b)
    return None if den==0 else float(a@b/den)

def main():
    OUT.mkdir(parents=True,exist_ok=False)
    signal.alarm(180);resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3))
    cfg=json.loads(PROTOCOL.read_text())
    save('protocol',cfg)
    save('start',dict(utc=datetime.now(timezone.utc).isoformat(),GPU=0,threads=1))
    global kernel_module
    spec=importlib.util.spec_from_file_location('bound_production_kernel',KERNEL)
    kernel_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(kernel_module)
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [KERNEL,PROTOCOL,Path(__file__)]}
    m=FiniteDiffusion(cfg);tests=[];initials=[];gradient_rows=[];stage_rows=[]
    def check(name,error,tol=1e-8):
        tests.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(error<tol)))
        save('tests',tests)
        assert error<tol,(name,error)
    # Independently compute each Bayes conditional, rather than call production twice.
    err=0.
    for t in range(1,m.T+1):
        a=m.alpha[t]/m.alpha[t-1]
        for x0 in [0,1]:
            prior=m.alpha[t-1]*x0+(1-m.alpha[t-1])*m.pi[m.f]
            px1=a+(1-a)*m.pi[m.f];px0=(1-a)*m.pi[m.f]
            numerator=prior*np.where(m.B>0,px1,1-px1)
            denominator=numerator+(1-prior)*np.where(m.B>0,px0,1-px0)
            err=max(err,float(np.max(abs(numerator/denominator-m.k[t][x0]))))
    check('independent_Bayes_vs_production',err)
    check('t1_clean_identity_zero',np.max(abs(m.k[1][0])))
    check('t1_clean_identity_one',np.max(abs(m.k[1][1]-1)))
    fixed=kernel_module.MarginalKernel([0.,1.],m.alpha)
    for t in range(1,m.T+1):check('fixed_support_'+str(t),np.max(abs(fixed.reverse([0.,1.],[0,1],[0,1],t-1,t)-[0,1])))
    rejected=False
    try:fixed.reverse([.1,1.],[0,1],[0,1],0,1)
    except ValueError:rejected=True
    check('invalid_fixed_support_rejected',0 if rejected else 1)
    for orbit in np.unique(m.orbits):check('type_permutation_cost_'+str(orbit),np.ptp(m.cost[m.orbits==orbit]))
    for seed in cfg['initializations']:
        rng=np.random.default_rng(seed);theta=rng.normal(0,.15,m.D);theta[:3]+=logit(m.pi)
        initials.append(theta);r=m.evaluate(theta)
        check('normalization_'+str(seed),max(np.max(abs(v.sum(1)-1)) for v in r['P'].values()))
        check('score_zero_'+str(seed),max(np.max(abs(v.sum(1))) for v in r['DP'].values()))
        check('policy_gradient_identity_'+str(seed),np.max(abs(r['g']-r['pg'])))
        check('constant_cost_pg_zero_'+str(seed),np.max(abs(r['constant_pg'])))
        eps=1e-5;fd=[]
        for d in range(m.D):
            delta=np.eye(m.D)[d]*eps
            fd.append((m.evaluate(theta+delta)['metrics']['terminal_cost']-m.evaluate(theta-delta)['metrics']['terminal_cost'])/(2*eps))
        check('finite_difference_'+str(seed),np.max(abs(np.array(fd)-r['g'])),1e-6)
        for method,key in [('exact','g'),('eager','eager'),('eager_mean_centered','eager_centered'),('single_step_clean','local'),('original_CE','ce')]:
            g=r[key];step=.001*g/max(np.linalg.norm(g),1e-15)
            after=m.evaluate(theta-step)['metrics']['terminal_cost']
            gradient_rows.append(dict(seed=seed,method=method,norm=float(np.linalg.norm(g)),
                cosine_to_exact=cos(g,r['g']),relative_error_to_exact=float(np.linalg.norm(g-r['g'])/max(np.linalg.norm(r['g']),1e-15)),
                equal_norm_step_terminal_change=after-r['metrics']['terminal_cost'],
                constant_cost_eager_norm=float(np.linalg.norm(r['eager_constant']))))
        for row in r['stages']:stage_rows.append(dict(seed=seed,**row))
        p=expit(m.features[2][7]@theta)
        probs=np.prod(np.where(m.B>0,p,1-p),axis=1)
        A=np.zeros((4,4))
        for v,(i,j) in zip(p,m.edges):A[i,j]=A[j,i]=v
        triangle=sum(A[i,j]*A[i,k]*A[j,k] for i,j,k in itertools.combinations(range(4),3))
        var=probs@(m.tri-triangle)**2
        check('soft_triangle_expectation_'+str(seed),abs(triangle-probs@m.tri))
        check('triangle_squared_risk_decomposition_'+str(seed),abs(probs@(m.tri-m.tri[m.refid])**2-(triangle-m.tri[m.refid])**2-var))
    save('gradient_comparison',gradient_rows);save('stage_gradients',stage_rows)
    save('gate',dict(passed=True,tests=len(tests),meaning='mathematical/implementation gate, not real model quality'))
    curves=[];finals=[];distributions=[];update_count=0
    for seed,initial in zip(cfg['initializations'],initials):
        for arm in cfg['trial']['arms']:
            theta=initial.copy()
            for step in range(41):
                r=m.evaluate(theta)
                if step in cfg['trial']['evaluations']:
                    curves.append(dict(seed=seed,arm=arm,step=step,**r['metrics']))
                if step==40:
                    finals.append(dict(seed=seed,arm=arm,**r['metrics']))
                    distributions.append(r['distribution']);break
                key={'exact_terminal_risk':'g','eager_terminal_cost_surrogate':'eager','reference_edge_CE':'ce'}[arm]
                gradient=r[key]
                assert np.isfinite(gradient).all()
                theta=theta-cfg['trial']['learning_rate']*gradient;update_count+=1
            save('curves',curves);save('finals',finals)
    for p,h in hashes.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    np.savez_compressed(OUT/'finite_distributions.npz',bits=m.B,cost=m.cost,blocks=m.blocks,
                        orbits=m.orbits,final_distributions=np.array(distributions))
    save('binding',dict(assets=hashes,external_code_executed=False))
    receipt=dict(status='complete_stopped',seconds=time.monotonic()-START,
                 RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                 CPU_threads=1,GPU_hours=0,formal_model_calls=0,formal_training_updates=0,
                 toy_arithmetic_parameter_updates=update_count,random_graphs=0,real_generated_graphs=0,
                 exact_graph_support=m.S,diffusion_steps=m.T,tests=len(tests),all_tests_passed=True,
                 new_long_training_authorized=False)
    save('receipt',receipt)
    print(json.dumps(dict(receipt=receipt,finals=finals,gradient_comparison=gradient_rows)))

if __name__=='__main__':
    try:main()
    except BaseException as e:
        if OUT.exists():save('failure',dict(error=repr(e),seconds=time.monotonic()-START))
        raise
