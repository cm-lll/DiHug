import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from worker import *
from scipy.optimize import minimize
def main():
    OUT.mkdir(exist_ok=False);p=json.loads((OLD/'protocol.json').read_text());bindings={};checks=[]
    rng=np.random.default_rng(2026111062)
    fixtures=[np.eye(5),np.zeros((5,7)),np.array([[1.,0],[-1.,0],[0,1],[0,-1],[0,0]]),np.ones((5,7))]+[rng.normal(size=(5,7)) for _ in range(12)]
    for i,G in enumerate(fixtures):
        H=G@G.T;coeff,cert=solver(H);v=coeff@G;mean=G.mean(0)
        result=minimize(lambda x:.5*np.sum((x-mean)**2),np.zeros(G.shape[1]),jac=lambda x:x-mean,constraints=[dict(type='ineq',fun=lambda x:G@x,jac=lambda x:G)],method='SLSQP',options=dict(ftol=1e-12,maxiter=300))
        assert result.success;assert np.max(abs(v-result.x))<2e-7;assert min(G@v)>=-1e-8
        checks.append(dict(case=i,maximum_primal_difference=float(abs(v-result.x).max()),KKT=cert))
    for path,h in json.loads((OLD/'bindings.json').read_text()).items():assert rt.sha(path)==h;bindings[path]=h
    for row in p['tasks']:
        d=OLD/f"{row['dataset']}_{row['seed']}"
        assert json.loads((d/'receipt.json').read_text())['status']=='completed'
        for path in [d/'evaluation_inputs.pt',d/'hybrid_complete.pt',d/'updates.json',d/'evaluation.json',*d.glob('eval_before*.npz'),*d.glob('eval_hybrid*.npz'),*d.glob('evaluation_scored_before*.npz'),*d.glob('evaluation_scored_hybrid*.npz'),d/'train_hybrid_u2001.npz']:bindings[str(path)]=rt.sha(path)
    for path in Path(__file__).parent.glob('*.py'):bindings[str(path)]=rt.sha(path)
    for n in ('audit.json','state_audit.json'):assert json.loads((OLD/n).read_text())['passed'];bindings[str(OLD/n)]=rt.sha(OLD/n)
    p.update(arms=['common'],baseline='reuse exact_feedback_updates hybrid mean; no baseline updates rerun',combination='project mean parameter gradient onto cone g_i dot v >=0; use existing rank-aware solver unmodified; deliver v to unchanged AdamW',planned_calls=dict(training_forwards=16,evaluation_forwards=24,VJPs=84,temporary_updates=16,reverse_chains=0),actual_update='record g_i dot actual AdamW displacement; no acceptance/rejection or optimizer-state reset; zero gradient still advances parent state',evaluation='reuse all previous six Xt and K256 uniform streams; development panel, not new independent confirmation',budgets=dict(wall_minutes_from_GPU_start=20,science_seconds_from_GPU_start=840,GPU_hours=.25,GPUs=1,CPU_threads=1,RSS_stop=None),numerical_checks=dict(existing_solver_unchanged=True,FP32_combined_cosine_floor=-1e-5,VJP_linearity_relative_tolerance=1e-3,cache_probability_tolerance=2e-6),no_claims=['common gradient is not necessarily common AdamW displacement','first-order estimates not finite-step guarantee','two terms remain sampled','warm-started from CE moments','not formal training or full-chain quality','evaluation reused development inputs'])
    rt.save(OUT/'protocol.json',p);rt.save(OUT/'bindings.json',bindings);rt.save(OUT/'cpu_gate.json',dict(passed=True,cases=checks));rt.save(OUT/'freeze.json',dict(epoch=time.time(),protocol_sha256=rt.sha(OUT/'protocol.json'),bindings_sha256=rt.sha(OUT/'bindings.json')))
    print(json.dumps(dict(output=str(OUT),CPU_solver_cases=len(checks),new_updates=16,new_forwards=40,new_VJPs=84)))
if __name__=='__main__':main()
