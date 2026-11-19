"""Read-only first-batch VJP audit. No optimizer or sampling entry is called."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import sys, json, time, traceback, importlib.util, csv
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/gdpo_feedback_repair_20261113'
PANEL = ROOT/'results/paper_ablation_20261112T190000'

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def main():
    allocation = float(os.environ['ALLOCATION_EPOCH'])
    stop = float(os.environ['SCIENCE_STOP_EPOCH'])
    s = module('audit_support', ROOT/'research_branches/one_step_vs_terminal_plan_20261111/update_preflight.py')
    w, torch, np = s.w, s.torch, s.np
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    def save(name, x): s.base.write(OUT/name, x)
    def check():
        if time.time() >= min(stop, allocation+1800): raise TimeoutError('science/GPU allocation limit')
    def group(n):
        if n.startswith('common_neighbor.'): return 'CN'
        if n.startswith('joint_head.'): return 'joint_head'
        if n.startswith('branch.'): return 'degree'
        if '.self_attn.' in n:
            return 'early_attention' if int(n.split('tf_layers.')[1].split('.')[0]) < 4 else 'late_attention'
        return 'other_backbone'
    calls = vjps = 0; results = []; sources = {}
    receipt = dict(status='running', allocation_epoch=allocation, GPU=0, optimizer_steps=0, new_graphs=0)
    try:
        p = json.loads((PANEL/'protocol.json').read_text())
        refs = json.loads((PANEL/'reference_metrics.json').read_text())
        for rel in ('research_branches/paper_ablation_20261112/worker.py', 'research_branches/paper_ablation_20261112/scoring.py', 'research_branches/one_step_vs_terminal_plan_20261111/gpu_preflight.py'):
            path = ROOT/rel; sources[str(path)] = s.base.sha(path)
            assert sources[str(path)] == p['bindings'][str(path)]
        save('environment.json', w.common.environment('cuda'))
        for job in p['jobs']:
            if job['architecture'] != 'A' or job['reward'] != 'typed': continue
            check(); row = p['parents'][job['parent']]; ds = job['dataset']
            directory = (ROOT/f'results/three_arm_learning_pilot_20261111T211641/{ds}_T_eager' if job['cached_complete'] else PANEL/job['id']/f'{ds}_T_eager')
            state_path = directory/'initial_complete.pt'
            parent = torch.load(row['readout'], map_location='cpu', weights_only=False)
            state = torch.load(state_path, map_location='cpu', weights_only=False)
            assert state['updates'] == 0 and not state['optimizer']['state']
            assert s.state_digest(state['model']) == s.state_digest(parent['model'])
            for path in (state_path, Path(row['readout']), Path(row['data'])):
                sources[str(path)] = s.base.sha(path)
                if str(path) in p['bindings']: assert sources[str(path)] == p['bindings'][str(path)]
            data = np.load(row['data']); edges = data['edges']; edges = edges.T if edges.shape[0] != 2 else edges
            case = w.common.make_case(data['types'], edges, tuple(map(tuple, data['schema'])))
            m = w.StructuralDenoiser(case, parent['binding']['config'], 'A').cuda(); m.load_state_dict(state['model'], strict=True); m.eval()
            rng = np.random.default_rng(); s.rng_restore(state['rng'], rng, 'cuda')
            before_rng = s.state_digest(s.rng_capture(rng, 'cuda')); before_model = s.state_digest(m.state_dict())
            ids, fam, query = w.common.plan_for(case, 2026111402, 1).block(0)
            params = list(m.parameters()); names = [n for n, _ in m.named_parameters()]
            accum = {key:[torch.zeros_like(q, dtype=torch.float64) for q in params] for key in ('H','N','actual')}
            batch = json.loads((directory/'round1_batch.json').read_text()); assert batch['policy_hash'] == before_model
            terms = []; records = []
            for plan in job['terminal_plan']['1']:
                folder = directory/f'train_round1_seed{plan["seed"]}'
                ident = json.loads((folder/'identity.json').read_text()); assert ident['model_hash'] == before_model and ident['updates'] == 0
                z = np.load(folder/'graph.npz'); counts = np.bincount(z['families'], minlength=case['n_families'])
                target_counts = np.array(refs[ds]['family_edges'])
                count = float(np.mean(((counts-target_counts)/np.maximum(1,target_counts))**2))
                terms.append(count)
            nc = np.array(terms); hc = np.array(batch['costs'])-nc
            na = 2*nc-nc.sum(); ha = 2*hc-hc.sum()
            for k, plan in enumerate(job['terminal_plan']['1']):
                folder = directory/f'train_round1_seed{plan["seed"]}'
                graph = np.load(folder/'graph.npz'); ee = graph['edges']; ee = ee.T if ee.shape[0] != 2 else ee
                edge_set = {tuple(sorted((int(u), int(v)))) for u,v in ee.T}
                bits = np.array([tuple(sorted((int(u),int(v)))) in edge_set for u,v in query.T])
                assert bits.sum() == len(edge_set)
                target = torch.as_tensor(bits, device='cuda')
                for t in plan['times']:
                    check(); tick = time.time(); path = folder/f'Xt_{t:04d}.npz'; sources[str(path)] = s.base.sha(path)
                    z = np.load(path); assert int(z['t']) == t
                    ee, ff = w.common.state_edges(case, z['xt'])
                    logits, prob = w.runtime.predict(m, case['space'].types, ee, ff, query, fam, t/1000, details=True); calls += 1
                    w.common.check_prob_device(prob, fam, case, m.prior)
                    lp = torch.log_softmax(logits.double(), dim=-1)
                    unweighted = .01*torch.where(target, lp[:,1], lp[:,0]).sum()/16
                    g = torch.autograd.grad(unweighted, params, retain_graph=True, allow_unused=True); vjps += 1
                    actual = torch.autograd.grad(unweighted*float(batch['advantages'][k]), params, allow_unused=True); vjps += 1
                    norm2 = 0.
                    for i,(gg, ag) in enumerate(zip(g, actual)):
                        if gg is None: assert ag is None; continue
                        assert torch.isfinite(gg).all() and torch.isfinite(ag).all()
                        gd = gg.double(); accum['H'][i].add_(gd, alpha=float(ha[k])); accum['N'][i].add_(gd, alpha=float(na[k])); accum['actual'][i].add_(ag.double())
                        norm2 += float(gd.square().sum())
                    records.append(dict(trajectory=k,t=t,unweighted_logscore=float(unweighted.detach()),basis_gradient_norm=norm2**.5,seconds=time.time()-tick))
                    save('progress.json',dict(job=job['id'],forwards=calls,VJPs=vjps,last=records[-1]))
                    del logits, prob, lp, unweighted, g, actual
            groups=[]
            for gr in sorted(set(map(group,names)))+['ALL']:
                ix=[i for i,n in enumerate(names) if gr=='ALL' or group(n)==gr]
                h2=n2=a2=dot=err=0.
                for i in ix:
                    h,n,a=accum['H'][i],accum['N'][i],accum['actual'][i]
                    h2+=float(h.square().sum());n2+=float(n.square().sum());a2+=float(a.square().sum());dot+=float((h*n).sum());err+=float((h+n-a).square().sum())
                groups.append(dict(group=gr,H_norm=h2**.5,N_norm=n2**.5,actual_norm=a2**.5,cos_H_N=dot/max((h2*n2)**.5,1e-300),linear_decomposition_relative_error=(err/max(a2,1e-300))**.5,mean_slot_norm=a2**.5/len(ids)))
            assert before_model == s.state_digest(m.state_dict())
            assert before_rng == s.state_digest(s.rng_capture(rng,'cuda'))
            assert all(q.grad is None for q in params)
            result=dict(job=job['id'],H_costs=hc.tolist(),N_costs=nc.tolist(),H_advantages=ha.tolist(),N_advantages=na.tolist(),active_slots=len(ids),groups=groups,records=records,model_unchanged=True,RNG_unchanged=True,grad_fields_untouched=True)
            save(job['id']+'.json', result); results.append(result)
            del accum, m, params, parent, state; torch.cuda.empty_cache()
        assert calls==64 and vjps==128
        receipt['status']='complete'
    except BaseException:
        receipt.update(status='failed',error=traceback.format_exc()); raise
    finally:
        receipt.update(forwards=calls,VJPs=vjps,completed_conditions=len(results),end_epoch=time.time(),conservative_GPU_hours=(time.time()-allocation)/3600,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20 if torch.cuda.is_initialized() else 0)
        save('receipt.json',receipt);save('sources.json',sources)

if __name__=='__main__': main()
