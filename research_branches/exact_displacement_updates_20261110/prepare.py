import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from worker import *
from scipy.optimize import minimize

def main():
    OUT.mkdir(exist_ok=False);p=json.loads((OLD/'protocol.json').read_text());bindings={};checks=[]
    random=np.random.default_rng(2026111063)
    fixtures=[np.eye(5),np.zeros((5,7)),np.array([[1.,0],[-1.,0],[0,1],[0,-1],[0,0]]),np.ones((5,7))]+[random.normal(size=(5,7)) for _ in range(12)]
    for i,G in enumerate(fixtures):
        d=random.normal(size=G.shape[1]);H=G@G.T;b=G@d;lam,cert=solver(H,b);v=d-lam@G
        other=minimize(lambda x:.5*np.sum((x-d)**2),np.zeros(len(d)),jac=lambda x:x-d,constraints=[dict(type='ineq',fun=lambda x:-G@x,jac=lambda x:-G)],method='SLSQP',options=dict(ftol=1e-12,maxiter=300))
        assert other.success;assert abs(v-other.x).max()<2e-7;assert max(G@v)<1e-8
        checks.append(dict(case=i,maximum_primal_difference=float(abs(v-other.x).max()),KKT=cert))
    torch.set_num_threads(1);torch.manual_seed(2026111063)
    parameter=torch.nn.Parameter(torch.randn(20));opt=torch.optim.AdamW([parameter],lr=2e-4,weight_decay=1e-12)
    parameter.grad=torch.randn(20);opt.step();prior=parameter.detach().clone();ema=prior.clone();G=random.normal(size=(5,20));parameter.grad=torch.tensor(G.mean(0),dtype=parameter.dtype)
    opt.step();proposal=parameter.detach().clone()-prior;ostamp=g.digest(rt.cpu_copy(opt.state_dict()));rstamp=g.digest(torch.get_rng_state());lam,cert=solver(G@G.T,G@proposal.double().numpy());ideal=proposal.double()-torch.tensor(lam@G)
    with torch.no_grad():parameter.copy_((prior.double()+ideal).float())
    actual=(parameter.detach()-prior).double().numpy();cos=G@actual/np.maximum(np.linalg.norm(G,axis=1)*np.linalg.norm(actual),1e-30)
    assert max(cos)<=1e-5;assert ostamp==g.digest(rt.cpu_copy(opt.state_dict()));assert rstamp==g.digest(torch.get_rng_state());ema=.99*ema+.01*parameter.detach()
    snapshot=dict(parameter=parameter.detach().clone(),optimizer=rt.cpu_copy(opt.state_dict()),EMA=ema,RNG=torch.get_rng_state(),updates=2)
    rt.checkpoint(OUT/'cpu_state_fixture.pt',snapshot);loaded=torch.load(OUT/'cpu_state_fixture.pt',weights_only=False);assert g.digest(snapshot)==g.digest(loaded)
    reference_next=torch.rand(7);torch.set_rng_state(loaded['RNG']);assert torch.equal(reference_next,torch.rand(7))
    for path,h in json.loads((OLD/'bindings.json').read_text()).items():assert rt.sha(path)==h;bindings[path]=h
    common=ROOT/'results/exact_common_updates_20261110'
    assert json.loads((common/'audit.json').read_text())['passed']
    for row in p['tasks']:
        d=OLD/f"{row['dataset']}_{row['seed']}";c=common/d.name
        for folder in (d,c):assert json.loads((folder/'receipt.json').read_text())['status']=='completed'
        paths=[d/'evaluation_inputs.pt',d/'hybrid_complete.pt',d/'updates.json',d/'evaluation.json',d/'train_hybrid_u2001.npz',c/'updates.json']
        for folder in (d,c):paths+=list(folder.glob('eval_*.npz'))+list(folder.glob('evaluation_scored_*.npz'))
        for path in paths:bindings[str(path)]=rt.sha(path)
    for folder in (OLD,common):
        for n in ('audit.json','model_summary.csv','evaluation_summary.csv','relations.csv','node_groups.csv'):bindings[str(folder/n)]=rt.sha(folder/n)
    for path in Path(__file__).parent.glob('*.py'):bindings[str(path)]=rt.sha(path)
    p.update(arms=['displacement'],baseline='reuse exact_feedback_updates hybrid and exact_common_updates common; no baseline updates rerun',combination='Mean of five parameter gradients to actual AdamW; project its complete displacement onto G*d<=0; no gradient projection, no extra acceptance probes',equation='min 0.5||d-d_Adam||^2 s.t. G*d<=0; d=d_Adam-G.T*lambda',planned_calls=dict(training_forwards=16,evaluation_forwards=24,VJPs=84,temporary_updates=16,reverse_chains=0),state_semantics='Commit corrected parameters, original proposal Adam moments/step, then EMA once; zero displacement still advances moments and EMA. Numerical/state failures invalidate partial boundary, not a resumable zero step.',evaluation='Reuse all six development Xt and paired K256 uniforms; raw only; not independent confirmation or full reverse generation',budgets=dict(wall_minutes_from_GPU_start=20,science_seconds_from_GPU_start=840,GPU_hours=.25,GPUs=1,CPU_threads=1,RSS_stop=None),numerical_checks=dict(written_displacement_max_cosine=1e-5,KKT_scaled_tolerance='1e-9*max(1,max(abs(b/scale)))',rank_cutoff=1e-12,no_ridge=True,VJP_linearity_relative_tolerance=1e-3,cache_probability_tolerance=2e-6),no_claims=['first-order estimates not finite-step guarantee','two objectives remain sampled','CE-trained parent Adam moments retained','parameter projection does not repair moment estimates','no formal training or full-chain efficacy','ordinary CUDA recomputation not bitwise certified'])
    rt.save(OUT/'protocol.json',p);rt.save(OUT/'bindings.json',bindings);rt.save(OUT/'cpu_gate.json',dict(passed=True,solver_cases=checks,FP32_actual_write_max_cosine=float(max(cos)),optimizer_unchanged_after_projection=True,EMA_after_corrected_parameters=True,RNG_unchanged_by_projection=True,save_load_state_exact=True,next_draw_exact=True));rt.save(OUT/'freeze.json',dict(epoch=time.time(),protocol_sha256=rt.sha(OUT/'protocol.json'),bindings_sha256=rt.sha(OUT/'bindings.json')))
    print(json.dumps(dict(output=str(OUT),CPU_solver_cases=len(checks),new_updates=16,new_forwards=40,new_VJPs=84)))
if __name__=='__main__':main()
