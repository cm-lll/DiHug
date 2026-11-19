"""CPU-only frozen assets, exact finite estimator tests and queue preparation."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import itertools, time
from pathlib import Path
from runtime import *
from objective import HardScore, loo_gradient, solve_gram
from exact_absent import ExactAbsentTriangles

def main():
    out = ROOT/'results/hard_clean_pilot_v2_20261110'
    out.mkdir(exist_ok=False)
    checks = []
    q = np.array(list(itertools.combinations(range(4), 2))).T
    typ = np.array([0,0,1,1]); fam = typ[q].sum(0)
    bits = np.array(list(itertools.product([0,1], repeat=6)), dtype=np.uint8)
    core = HardScore(typ,q,fam,np.array([1,1,0,1,0,1]),3)
    costs = core.batch(bits); p = np.linspace(.15,.8,6)
    prob = np.prod(np.where(bits,p,1-p),1)
    exact = (costs*prob[:,None]).T @ (bits-p)
    estimated = np.zeros_like(exact)
    for i in range(64):
        # Exhaustive expectation over two INDEPENDENT draws, not finite-sample quality.
        for j in range(64):
            estimated += prob[i]*prob[j]*loo_gradient(p,bits[[i,j]],costs[[i,j]])
    analytic = ExactAbsentTriangles(core)
    value, grad = analytic.value_gradient(p)
    assert abs(value-float(prob@costs[:,3])) < 1e-12
    assert abs(grad-exact[3]).max() < 1e-12
    checks.append(dict(test='exact_absent_triangle_second_moment_and_gradient',value=value,error=float(abs(grad-exact[3]).max())))
    error = float(abs(estimated-exact).max()); assert error < 1e-12
    checks.append(dict(test='LOO_K2_exact_unbiased_margin_gradient',error=error))
    for i,bits0 in enumerate(bits):
        v = core.features(bits0); a=np.zeros((4,4),int); a[q[0],q[1]]=bits0; a+=a.T
        tri=sum(a[i,j]*a[i,k]*a[j,k] for i,j,k in itertools.combinations(range(4),3))
        assert v['triangles'].sum() == tri
        assert v['degree'].sum() == 2*bits0.sum()
    checks.append(dict(test='all_64_triangle_degree_counts',passed=True))
    for h in (np.eye(3), np.array([[1.,-1.],[-1.,1.]]), np.ones((3,3)), np.zeros((3,3))):
        lam,res = solve_gram(h); checks.append(dict(test='gram_KKT',residual=res,coefficients=lam.tolist()))
    models=[]; bindings={}
    for ds,seeds in [('IMDB',[2026110411,2026110412]),('PubMed',[2026110421,2026110422])]:
        for seed in seeds:
            base=ROOT/f'results/cn_multiseed_20261104T042546/run/{ds}_{seed}_A_late'
            cp=base/'checkpoint_u2000.pt'; z=torch.load(cp,map_location='cpu'); cfg=z['binding']['config']
            h=sha(cp); assert h==json.loads((base/'checkpoint_u2000.sha256.json').read_text())['sha256']
            data=g.SCI/'package'/cfg['datasets'][ds]['file']; dh=sha(data)
            assert dh==cfg['datasets'][ds]['sha256']
            assert z['updates']==z['ema']['updates']==2000 and z['binding']['arm']=='A'
            assert cfg['seed']==seed and z['binding']['train_seed']==seed
            history=base/'training.jsonl'
            for pth in (cp,data,history):bindings[str(pth)]=sha(pth)
            models.append(dict(dataset=ds,seed=seed,checkpoint=str(cp),checkpoint_sha256=h,data=str(data),data_sha256=dh,history=str(history)))
    tasks=[]
    for i,row in enumerate(models):
        for arm in ('CE','HARD_MGDA'):
            tasks.append(dict(row,arm=arm,gpu=len(tasks),task=f"{row['dataset']}_{row['seed']}_{arm}"))
    # Current source inventory only; no historical file is edited or copied over.
    sources = list(HERE.glob('*.py')) + list(g.ROOT.glob('*.py')) + list((g.ROOT/'experiments/o_cn_matched_execute_v1').glob('*.py')) + list(g.SCI.glob('*.py')) + list((g.SCI/'package').rglob('*.py'))
    for path in sources:bindings[str(path)]=sha(path)
    cfg=dict(version='hard_clean_real_pilot_v2_exact_absent_triangle',created_epoch=time.time(),tasks=tasks,
             updates_each=128,parent_updates=2000,train_hard_draws=16,evaluation_hard_draws=32,
             evaluation_times=[1,75,250,550,800,1000],evaluation_offsets=[0,32,128],readouts=['raw','EMA0.99'],
             auxiliary_seed_offset=730001,evaluation_seed_offset=810001,
             estimator_revision='same hard risk; absent typed triangles exact second moment, remaining four LOO; no threshold/weight/objective change',
             parent_window=str(ROOT/'results/hard_clean_pilot_20261110/window.json'),
             main_inputs_per_update=1,loss='CE control or pure five-group hard risk with LOO then MGDA; no CE in structural arm',
             objective_groups=HardScore.names,triangle_groups='schema-feasible triples grouped by reference count >0 versus zero',
             solver='raw parameter-gradient Gram simplex minnorm; same samples; no gradient normalization or step matching',
             optimizer='parent AdamW including moments; no reset, clipping, LR change or candidate rejection',
             zero_sample_gradient='keep zero gradients in Gram; if combined norm <=1e-12 stop affected trajectory and report estimator limitation',
             interpretation='warm-start 2000->2128 local training pilot, not from-scratch method comparison or full-generation evidence',
             budgets=dict(wall_minutes=90,science_minutes=60,GPU_hours=8,max_GPUs=8,worker_minutes=45),
             gate=dict(CPU_checks=True,load_roundtrip_exact=True,next_input_and_aux_RNG_exact=True,finite_gradient=True,
                       historical_input_hash_exact=True,throughput_panel_safety_margin=1.5,
                       quality_gate=False,recomputed_FP32_bitwise_required=False),
             no_full_generation=True,no_automatic_long_training=True)
    save(out/'protocol.json',cfg);save(out/'bindings.json',bindings);save(out/'cpu_gate.json',dict(passed=True,checks=checks,models=len(models)))
    print(json.dumps(dict(output=str(out),models=len(models),tasks=len(tasks),bindings=len(bindings),CPU_checks=checks)))
if __name__=='__main__':main()

