"""Two frozen parents, paired CE/PS local trajectories; never formal training."""
import os, sys, json, time, signal, traceback, importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ['OBJECTIVE_DIAG_OUT'])
spec = importlib.util.spec_from_file_location('old_bounded_worker', ROOT/'results/output_boundary_replay_20261108T081654/worker.py')
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)
sys.path.insert(0, str(ROOT/'research_branches/reproducible_aggregation_20261108'))
from adapter import execution_policy, VERSION
g, fr, torch, np = w.g, w.fr, w.torch, w.np
protocol = w.read(OUT/'protocol.json')
begin = time.time()
stop = False
status = dict(status='loading', start_epoch=begin, updates=0, forwards=0, completed_branches=[], gates=[])

def request_stop(*args):
    global stop
    stop = True

for sig in (signal.SIGTERM, signal.SIGINT):
    signal.signal(sig, request_stop)

def check():
    if stop or time.time() >= min(protocol['science_stop_epoch'], begin+870):
        raise TimeoutError('bounded diagnostic stop')

def save(name, value):
    fr.save(OUT/name, value)

def main():
    fr.setup()
    fr.authenticate_sources()
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert '2080 Ti' in torch.cuda.get_device_name(0)
    save('environment.json', dict(**g.common.environment('cuda'), execution_policy=VERSION))
    updates, probes, moves, bindings = [], [], [], []
    tasks = w.read(ROOT/'results/output_boundary_replay_20261108T081654/tasks.json')
    for task in tasks:
        check()
        ds, cp = task['dataset'], Path(task['checkpoint'])
        assert w.sha(cp) == task['checkpoint_sha256']
        parent = torch.load(cp, map_location='cpu')
        cfg, binding = task['row']['config'], parent['binding']
        assert parent['updates'] == task['start'] and binding['config'] == cfg
        assert binding['dataset'] == ds and binding['train_seed'] == task['seed'] and binding['arm'] == 'A_PS'
        data = ROOT/g.SCI.relative_to(g.ROOT)/'package'/cfg['datasets'][ds]['file']
        assert w.sha(data) == cfg['datasets'][ds]['sha256']
        with np.load(data, allow_pickle=False) as z:
            case = g.common.make_case(z['types'], z['edges'].T, tuple(map(tuple, z['schema'])))
        m = g.StructuralDenoiser(case, cfg, 'A').cuda()
        o = torch.optim.AdamW(m.parameters(), **{k:v for k,v in cfg['optimizer'].items() if k!='type'})
        e, r = g.ParameterEMA(m), np.random.default_rng(1)
        ps = w.make_objective(case, cfg, ds, 'PS')
        assert ps.objective_binding() == binding['output_objective']['target']
        history = [json.loads(x) for x in (w.OLD/'run'/task['task']/'training.jsonl').read_text().splitlines()]
        b = None
        def restore():
            nonlocal b
            work = w.cpu_copy(parent)
            g.restore_payload(work,m,o,e,work['binding'],lambda x:g.rng_restore(x,r,'cuda'))
            m.train(); o.zero_grad(set_to_none=True)
            b = g.TrainingBoundary(m,o,e,r,'cuda',parent['updates'])
            live = b.checkpoint(binding)
            for k in ['model','optimizer','ema','rng','updates']:
                assert g.digest(live[k]) == g.digest(parent[k]), 'restore '+k

        fixed = []
        for panel in ['historical','new_diagnostic']:
            for t in protocol['times']:
                seed = (w.seed32(task['seed'], purpose='fixed_validation_noise',time=t) if panel=='historical'
                        else w.seed32(protocol['new_probe_seed'][ds],purpose='reproducible_objective_probe_v1',time=t))
                inp = w.pair_inputs(case,cfg,10000+t,np.random.default_rng(seed),t)[0]
                fixed.append((panel,t,seed,inp))
        oldval = next(x for x in w.read(w.OLD/'run'/task['task']/'validation.json') if x['updates']==task['start'])
        oldhashes = {x['t']:x['input_sha256'] for x in oldval['raw']['rows']}
        input_binding = []
        for panel,t,seed,inp in fixed:
            h = w.input_hash([inp])
            if panel=='historical': assert h == oldhashes[t]
            input_binding.append(dict(panel=panel,t=t,seed=seed,input_sha256=h,
                                      duplicates_historical_input=panel!='historical' and h in oldhashes.values()))
        torch.save(dict(inputs=fixed, purpose='diagnostic, not independent final test'),OUT/f'{ds}_fixed_inputs.pt')
        bindings.append(dict(dataset=ds,train_seed=task['seed'],parent=str(cp),parent_sha256=w.sha(cp),
                             data_sha256=w.sha(data),start=task['start'],inputs=input_binding))
        save('asset_binding.json',bindings)

        def evaluate(arm,offset,readout='raw'):
            check()
            state_before = g.digest(b.checkpoint(binding))
            mode = m.training
            with b.validation():
                raw = w.cpu_copy(m.state_dict()) if readout=='EMA0.99' else None
                try:
                    if raw is not None: m.load_state_dict(e.readout_state_dict(m),strict=True)
                    m.eval()
                    for panel,t,seed,inp in fixed:
                        check()
                        token = ('objective_diag',ds,arm,offset,readout,panel,t)
                        ctx = g.FrozenTypedContext(m,case['space'].types,inp['edges'],inp['edge_families'],token)
                        try:
                            with torch.no_grad():
                                graph,logits,p = ctx.predict(inp['query'],inp['f'],t/1000,token,details=True)
                                g.common.check_prob_device(p,inp['f'],case,m.prior)
                                ce = w.supported_binary_ce(logits,torch.as_tensor(inp['labels'],device='cuda'),m.prior[graph.query_families]).cpu().numpy()
                                prob = p.cpu().numpy()
                            assert np.isfinite(ce).all() and np.isfinite(prob).all()
                            y = inp['labels']; rel = []
                            for f in range(case['n_families']):
                                ix = inp['f']==f; pos = ix&(y==1); neg = ix&(y==0)
                                pred = np.zeros(len(case['space'].types)); target = np.zeros_like(pred)
                                for ends in inp['query']:
                                    np.add.at(pred,ends[ix],prob[ix]); np.add.at(target,ends[ix],y[ix])
                                nodes = np.isin(case['space'].types,case['space'].pairs[f])
                                rel.append(dict(family=f,types=list(map(int,case['space'].pairs[f])),slots=int(ix.sum()),positive_count=int(pos.sum()),negative_count=int(neg.sum()),
                                    positive_CE=float(ce[pos].mean()) if pos.any() else None,negative_CE=float(ce[neg].mean()) if neg.any() else None,
                                    expected_edges=float(prob[ix].sum()),true_edges=int(pos.sum()),missed_positive_mass=float((1-prob[pos]).sum()),false_positive_mass=float(prob[neg].sum()),
                                    expected_relation_degree_MAE=float(abs(pred[nodes]-target[nodes]).mean()) if nodes.any() else None))
                            row=dict(dataset=ds,seed=task['seed'],arm=arm,offset=offset,completed=b.completed,readout=readout,panel=panel,t=t,input_sha256=w.input_hash([inp]),
                                     duplicates_historical_input=panel!='historical' and w.input_hash([inp]) in oldhashes.values(),
                                     CE=float(ce.sum()/case['space'].total),positive_CE=float(ce[y==1].mean()),negative_CE=float(ce[y==0].mean()),expected_edges=float(prob.sum()),relations=rel)
                            probes.append(row); status['forwards']+=1
                            np.savez_compressed(OUT/f'{ds}_{arm}_{offset}_{readout}_{panel}_{t}.npz',probability=prob,CE=ce)
                        finally: ctx.close()
                finally:
                    if raw is not None: m.load_state_dict(raw,strict=True)
                    m.train(mode)
            assert g.digest(b.checkpoint(binding)) == state_before, 'evaluation state/RNG contamination'
            save('probes.json',probes)

        class Objective:
            def __init__(self,kind): self.kind=kind; self.last={}
            def weights(self,q,f,inp,*,device,dtype): return torch.ones(len(f),device=device,dtype=dtype)
            def __call__(self,p,q,f,inp,*,logits):
                return (p.sum()*0,p.new_empty(0)) if self.kind=='CE' else ps(p,q,f,inp,logits=logits)

        with execution_policy(True):
            restore(); evaluate('shared_start',0)
            for kind in ['CE','PS']:
                restore(); obj=Objective(kind)
                for offset in range(1,protocol['steps']+1):
                    check(); u=b.completed+1
                    before={n:p.detach().cpu().clone() for n,p in m.named_parameters()} if offset in [1,32] else None
                    torch.cuda.synchronize(); tick=time.perf_counter()
                    with b.step(u):
                        inp=g.one_input(case,cfg,u,r); expected=history[u-1]
                        ih=w.input_hash([inp]); assert ih==expected['input_sha256']
                        assert g.digest(g.rng_capture(r,'cuda'))==expected['full_rng_before']
                        rec=w.update(m,o,case,inp,1000,'cuda',e,u-1,transaction=b,objective=obj)
                        assert g.digest(g.rng_capture(r,'cuda'))==expected['full_rng_after']
                    torch.cuda.synchronize()
                    status['updates']+=1
                    updates.append(dict(dataset=ds,seed=task['seed'],arm=kind,offset=offset,update=u,t=inp['t'],input_sha256=ih,seconds=time.perf_counter()-tick,
                                        **{k:rec[k] for k in ['loss','unweighted_CE','auxiliary_loss','gradient_norm']}))
                    if before is not None:
                        delta={n:p.detach().cpu()-before[n] for n,p in m.named_parameters()}
                        moves.append(dict(dataset=ds,arm=kind,offset=offset,groups={c:w.stats(torch.cat([v.flatten() for n,v in delta.items() if c=='all' or w.category(n)==c])) for c in ['all','CN_exclusive','joint_head','degree_branch','shared_backbone_other']}))
                        torch.save(delta,OUT/f'{ds}_{kind}_{offset}_parameter_displacement.pt')
                        del before,delta
                    if offset==1:
                        reference=torch.load(ROOT/f'results/reproducible_aggregation_20261108T113027/attempt3/{ds}_{kind}_strict_first.pt',map_location='cpu')['state']
                        live=b.checkpoint(binding)
                        checks={k:g.digest(live[k])==g.digest(reference[k]) for k in reference}
                        status['gates'].append(dict(dataset=ds,arm=kind,first_step_exact_against_previous=checks))
                        assert all(checks.values()), 'strict first step changed since preceding ticket'
                        del reference,live
                    if offset in protocol['evaluation_offsets']: evaluate(kind,offset)
                    save('updates.json',updates); save('progress.json',status)
                evaluate(kind,protocol['steps'],'EMA0.99')
                fr.atomic_checkpoint(OUT/f'{ds}_{kind}_diagnostic_state.pt',dict(diagnostic_only=True,local_objective=kind,parent_sha256=w.sha(cp),execution_policy=VERSION,state=b.checkpoint(binding)))
                status['completed_branches'].append(ds+'_'+kind)
        assert w.sha(cp)==task['checkpoint_sha256']
        del m,o,e,b,parent,ps
        torch.cuda.empty_cache()
    save('parameter_moves.json',moves)
    fr.authenticate_sources()
    assert status['updates']==128 and status['forwards']==108
    status['status']='complete'

if __name__=='__main__':
    try: main()
    except BaseException as exc:
        status.update(status='stopped_or_failed',error=repr(exc),traceback=traceback.format_exc())
    finally:
        status.update(end_epoch=time.time(),seconds=time.time()-begin,peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20 if torch.cuda.is_initialized() else 0)
        save('receipt.json',status)
        print(json.dumps(status),flush=True)
    raise SystemExit(0 if status['status']=='complete' else 2)
