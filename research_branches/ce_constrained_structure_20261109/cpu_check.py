"""Exact 4-node typed-graph enumeration and constrained optimization checks.

No torch, GPU, denoiser, real-graph sampling, or production training. All 64
simple graphs are exhaustively enumerated, not sampled. Finite-dimensional
optimization is a numerical mechanism test, not an experimental quality result.
"""
import csv
import hashlib
import itertools
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.special import expit

from controller import Controller

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/ce_constrained_structure_20261109'
sys.path.insert(0, str(ROOT/'research_branches/direct_structure_cpu_20261108'))
from score import DirectScore

CONFIG = dict(initialization_seeds=[2026110901,2026110902,2026110903],
              float_dtype='float64', primal_rate=0.02, dual_rate=0.05,
              iterations=4000, checkpoint_every=100, fixed_joint_lambda=0.001,
              finite_difference_epsilon=1e-5, gradient_tolerance=2e-7,
              oracle_maxiter=500, oracle_ftol=1e-10, budget_relative_slack=0.,
              seconds_limit=600, bytes_limit=2*1024**3,
              scope='Exact CPU toy checks only; no production-ready optimizer claim')

class Panel:
    def __init__(self, name):
        self.name = name
        self.types = np.array([0,0,1,1])
        self.q = np.array(list(itertools.combinations(range(4),2))).T
        self.f = np.array([0,1,1,1,1,2])
        self.y = np.array([1,1,0,1,0,1] if name=='triangle_tail' else [1,1,1,0,0,0], dtype=float)
        self.bits = np.array(list(itertools.product([0.,1.],repeat=6)))
        self.scores = np.array([DirectScore(self.types,self.q,self.f,self.y,b,3).base for b in self.bits])
        # Three fixed hard states; features depend on state/schema, never labels.
        states = np.array([[1,0,0,1,0,0], [0,1,1,0,1,0], [1,1,0,1,1,1]])
        features = []
        for j,b in enumerate(states):
            adj=np.zeros((4,4));adj[self.q[0],self.q[1]]=b;adj+=adj.T
            deg=adj.sum(1)
            features.append(np.column_stack([np.ones(6),np.eye(3)[self.f],
                (deg[self.q[0]]+deg[self.q[1]])/6,
                (adj@adj)[self.q[0],self.q[1]]/2,
                np.full(6,j/2)]))
        self.x=np.array(features)

    def evaluate(self, theta, free=False):
        z = theta.reshape(3,6) if free else self.x@theta
        p=expit(z)
        ce=float(np.mean(np.logaddexp(0,z)-self.y*z))
        cg=(p-self.y)/18
        scores=[];sg=np.empty_like(p)
        for j in range(3):
            # Stable product probabilities for exact full support, including near boundaries.
            log_mass=np.sum(self.bits*(-np.logaddexp(0,-z[j]))+(1-self.bits)*(-np.logaddexp(0,z[j])),axis=1)
            mass=np.exp(log_mass)
            assert abs(mass.sum()-1)<1e-12
            scores.append(float(mass@self.scores))
            sg[j]=np.sum(mass[:,None]*self.scores[:,None]*(self.bits-p[j]),axis=0)/3
        if free:
            return ce,float(np.mean(scores)),cg.ravel(),sg.ravel()
        return ce,float(np.mean(scores)),np.einsum('se,sek->k',cg,self.x),np.einsum('se,sek->k',sg,self.x)

def write_csv(path, rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    OUT.mkdir(exist_ok=True,parents=True)
    # Freeze protocol before evaluating candidate outcomes.
    (OUT/'cpu_protocol.json').write_text(json.dumps(CONFIG,indent=2))
    started=time.monotonic()
    resource.setrlimit(resource.RLIMIT_AS,(CONFIG['bytes_limit'],CONFIG['bytes_limit']))
    checks=[];results=[];curves=[]
    def deadline():
        if time.monotonic()-started>CONFIG['seconds_limit']:
            raise TimeoutError('CPU budget reached; no automatic extension.')
    for name in ['triangle_tail','star']:
        panel=Panel(name)
        # Identity: full-graph score derivative equals expectation of the edge
        # conditional finite difference estimator; no Monte Carlo uncertainty.
        z=np.array([-2.,-.7,.1,.6,1.2,2.]);p=expit(z)
        mass=np.prod(np.where(panel.bits,p,1-p),axis=1)
        enumerated=np.sum(mass[:,None]*panel.scores[:,None]*(panel.bits-p),axis=0)
        edge=np.zeros(6)
        for b,m in zip(panel.bits,mass):
            edge+=m*DirectScore(panel.types,panel.q,panel.f,panel.y,b,3).logit_estimate(p)
        err=float(np.max(abs(enumerated-edge)));assert err<1e-12
        checks.append(dict(test='edge_difference_identity',case=name,max_abs_error=err,passed=True))
        # The reference itself gives zero score: constrained reformulation alone
        # supplies no diversity requirement.
        ri=np.flatnonzero(np.all(panel.bits==panel.y,axis=1)).item()
        assert panel.scores[ri]==0.
        for seed in CONFIG['initialization_seeds']:
            theta0=np.random.default_rng(seed).normal(0,.4,size=7)
            ce0,s0,cg,sg=panel.evaluate(theta0)
            fd=[]
            for k in range(7):
                d=np.zeros(7);d[k]=CONFIG['finite_difference_epsilon']
                a,b=panel.evaluate(theta0+d),panel.evaluate(theta0-d)
                fd.append([(a[0]-b[0])/(2*d[k]),(a[1]-b[1])/(2*d[k])])
            err=float(np.max(abs(np.array(fd)-np.stack([cg,sg],axis=1))))
            assert err<CONFIG['gradient_tolerance']
            checks.append(dict(test='analytic_gradient_finite_difference',case=f'{name}_{seed}',max_abs_error=err,passed=True))
            for arm in ['CE','fixed_joint','constrained']:
                theta=theta0.copy();dual=Controller(ce0,s0,CONFIG['dual_rate'])
                resumed=None
                for it in range(1,CONFIG['iterations']+1):
                    if it%100==0:deadline()
                    ce,s,cg,sg=panel.evaluate(theta)
                    if arm=='CE': grad=cg
                    elif arm=='fixed_joint': grad=cg+CONFIG['fixed_joint_lambda']*sg
                    else:
                        a,b=dual.coefficients();grad=a*cg+b*sg
                    theta-=CONFIG['primal_rate']*grad
                    post=panel.evaluate(theta)
                    if arm=='constrained':dual.commit(post[0])
                    if it==2000:
                        # Roundtrip includes all state of this CPU solver; it is
                        # not a substitute for production AdamW/EMA/RNG recovery.
                        restored=Controller.from_state_dict(json.loads(json.dumps(dual.state_dict())))
                        tt=theta.copy();_,_,cc,ss=panel.evaluate(tt)
                        aa,bb=restored.coefficients()
                        if arm=='constrained':
                            expected=theta-CONFIG['primal_rate']*(dual.coefficients()[0]*cc+dual.coefficients()[1]*ss)
                            got=tt-CONFIG['primal_rate']*(aa*cc+bb*ss)
                            assert np.array_equal(expected,got)
                            checks.append(dict(test='CPU_controller_roundtrip_next_update',case=f'{name}_{seed}',max_abs_error=0.,passed=True))
                    if it%CONFIG['checkpoint_every']==0:
                        curves.append(dict(case=name,seed=seed,arm=arm,iteration=it,CE=post[0],structure=post[1],violation=post[0]/ce0-1,multiplier=dual.multiplier))
                results.append(dict(case=name,seed=seed,arm=arm,CE_start=ce0,structure_start=s0,
                    CE_final=post[0],structure_final=post[1],relative_CE_violation=post[0]/ce0-1,
                    multiplier=dual.multiplier,iterations=CONFIG['iterations'],solver_success=True,
                    feasible=bool(post[0]<=ce0*(1+1e-6))))
            # Separate numerical reference solver, not the proposed production method.
            oracle=minimize(lambda x:panel.evaluate(x)[1]/s0,theta0,
                jac=lambda x:panel.evaluate(x)[3]/s0,method='SLSQP',
                constraints=[dict(type='ineq',fun=lambda x:1-panel.evaluate(x)[0]/ce0,
                                  jac=lambda x:-panel.evaluate(x)[2]/ce0)],
                options=dict(maxiter=CONFIG['oracle_maxiter'],ftol=CONFIG['oracle_ftol']))
            ce,s,_,_=panel.evaluate(oracle.x)
            results.append(dict(case=name,seed=seed,arm='local_SLSQP_reference',CE_start=ce0,structure_start=s0,
                CE_final=ce,structure_final=s,relative_CE_violation=ce/ce0-1,multiplier=None,
                iterations=int(oracle.nit),solver_success=bool(oracle.success),feasible=bool(ce<=ce0*(1+1e-6))))
            write_csv(OUT/'toy_results.csv',results);write_csv(OUT/'toy_curves.csv',curves)
            (OUT/'checks.json').write_text(json.dumps(checks,indent=2))
    # Sign, zero-reference and failure state behavior tests.
    c=Controller(1.,1.);c.commit(2.);assert c.multiplier>1
    c.commit(0.);assert c.multiplier==1
    before=c.state_dict()
    try:c.commit(float('nan'));raise AssertionError('missing nonfinite rejection')
    except FloatingPointError:assert c.state_dict()==before
    try:Controller(0.,1.);raise AssertionError('zero reference accepted')
    except ValueError:pass
    checks.append(dict(test='dual_sign_and_invalid_input_state',case='controller',max_abs_error=0.,passed=True))
    (OUT/'checks.json').write_text(json.dumps(checks,indent=2))
    sources=[Path(__file__),Path(__file__).with_name('controller.py'),ROOT/'research_branches/direct_structure_cpu_20261108/score.py',ROOT/'research_branches/hard_structure_edge_direction_cpu_20261108/metrics.py']
    summary=dict(status='CPU_complete',wall_seconds=time.monotonic()-started,
                 peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                 GPU_hours=0,production_model_calls=0,production_updates=0,full_generation_chains=0,
                 exact_toy_graphs=128,checked_cases=6,checks=len(checks),
                 numpy=np.__version__,scipy=scipy.__version__,python=sys.executable,
                 source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                 limitations=['Nonconvex shared parameter toy; no global optimality guarantee.',
                    'Exact post-update CE is available here; stochastic production dual update remains unvalidated.',
                    'SLSQP is a local reference, not a replacement optimizer or a global oracle.',
                    'No GPU devices available; no production integration or training launched.'])
    (OUT/'cpu_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
