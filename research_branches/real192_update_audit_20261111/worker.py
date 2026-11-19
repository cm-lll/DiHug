"""Bounded real-graph instrumentation of the frozen B4K16 update.

No new objective, clipping, optimizer reset, or reverse generation. Component
VJPs read the already-built graph and do not accumulate into parameter.grad.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import sys, time, json, signal, traceback, importlib.util, inspect
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'results/real192_update_audit_20261111'
PREV = ROOT / 'results/single_step_formal_resume_20261111'
sys.path.insert(0, str(ROOT / 'research_branches/single_step_formal_resume_20261111'))
spec = importlib.util.spec_from_file_location('frozen_single_step', ROOT / 'research_branches/single_step_formal_resume_20261111/worker.py')
base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
rt, g, torch, np = base.rt, base.g, base.torch, base.np
NAMES = ['count', 'joint_degree', 'present_triangles', 'absent_triangles', 'components']
PROBES = {2001, 2008, 2010, 2012, 2017, 2046, 2050}


def read(path):
    return json.loads(Path(path).read_text())


def check():
    if time.time() >= read(OUT / 'window.json')['science_stop_epoch']:
        raise TimeoutError('science stop')


def dot(a, b):
    return float((a.double() * b.double()).sum())


def cosine(a, b):
    denominator = dot(a, a) ** .5 * dot(b, b) ** .5
    return dot(a, b) / denominator if denominator else None


class Tracker:
    def __init__(self, model, folder):
        self.params = list(model.parameters())
        self.names = [n for n, _ in model.named_parameters()]
        self.folder = folder
        self.groups = {}
        offset = 0
        for name, p in model.named_parameters():
            self.groups.setdefault(base.old.group(name), []).append((offset, offset + p.numel()))
            offset += p.numel()
        self.size = offset
        self.extra_vjps = 0
        self.sum_checks = []

    def flat(self, values):
        return torch.cat([(torch.zeros_like(p) if v is None else v).detach().flatten()
                          for p, v in zip(self.params, values)])

    def begin(self, m, o, u):
        self.u = u
        self.theta = self.flat(self.params)
        self.parts = torch.zeros((5, self.size), device='cuda') if u in PROBES else None
        self.output_rows = []
        self.stats = {}

    def feedback(self, margin, feedback, probability, B, i, t):
        self.output_rows.append(dict(index=i, t=t, expected_edges=float(probability.sum()),
                                     below_1e8=int((probability < 1e-8).sum()),
                                     above_1_minus_1e8=int((probability > 1-1e-8).sum()),
                                     feedback_norms=np.linalg.norm(feedback, axis=1).tolist()))
        if self.parts is not None:
            original = [p.grad.detach().clone() if p.grad is not None else None for p in self.params]
            for k in range(5):
                gradients = torch.autograd.grad(margin, self.params,
                    grad_outputs=torch.as_tensor(feedback[k] / (5 * B), device='cuda', dtype=margin.dtype),
                    retain_graph=True, allow_unused=True)
                self.parts[k].add_(self.flat(gradients))
                self.extra_vjps += 1
            assert all((a is None and p.grad is None) or
                       (a is not None and p.grad is not None and torch.equal(a, p.grad))
                       for a, p in zip(original, self.params)), 'probe overwrote training gradients'

    def pre_step(self, m, o):
        self.gradient = self.flat([p.grad for p in self.params])
        assert torch.isfinite(self.gradient).all()
        # Attribute the realized update only by local dot products, not additive
        # AdamW counterfactuals. Moment magnitudes are recorded independently.
        moment = second = 0.
        steps = []
        for p in self.params:
            state = o.state.get(p, {})
            if 'exp_avg' in state:
                moment += float(state['exp_avg'].double().square().sum())
                second += float(state['exp_avg_sq'].double().sum())
                steps.append(float(state['step']))
        self.stats.update(before_moment_norm=moment ** .5, before_second_moment_sum=second,
                          optimizer_steps_before=sorted(set(steps)))
        if self.parts is not None:
            combined = self.parts.sum(0)
            difference = (combined - self.gradient).double()
            relative = float(difference.norm()) / max(float(self.gradient.double().norm()), 1e-30)
            self.stats.update(component_sum_relative_error=relative,
                              component_sum_max_abs=float(difference.abs().max()))
            self.sum_checks.append(relative)
            if relative > .005:
                raise RuntimeError('component VJP sum differs from actual accumulated gradient')

    def finish(self, m, o):
        actual = self.flat(self.params) - self.theta
        assert torch.isfinite(actual).all()
        gg, dd = dot(self.gradient, self.gradient), dot(actual, actual)
        groups = {}
        for name, spans in self.groups.items():
            ix = torch.cat([torch.arange(a, b, device='cuda') for a, b in spans])
            gr, de = self.gradient[ix], actual[ix]
            entry = dict(gradient_norm=dot(gr, gr) ** .5,
                         displacement_norm=dot(de, de) ** .5,
                         gradient_dot_displacement=dot(gr, de))
            if self.parts is not None:
                entry['components'] = {k: dict(gradient_norm=dot(v[ix], v[ix]) ** .5,
                                               dot_displacement=dot(v[ix], de))
                                       for k, v in zip(NAMES, self.parts)}
            groups[name] = entry
        self.stats.update(update=self.u, gradient_norm=gg ** .5, displacement_norm=dd ** .5,
                          parameter_norm_before=dot(self.theta, self.theta) ** .5,
                          gradient_dot_displacement=dot(self.gradient, actual),
                          gradient_displacement_cosine=cosine(self.gradient, actual),
                          groups=groups, input_probabilities=self.output_rows)
        if self.parts is not None:
            self.stats['components'] = {k: dict(gradient_norm=dot(v, v) ** .5,
                                                dot_displacement=dot(v, actual),
                                                cosine_displacement=cosine(v, actual))
                                        for k, v in zip(NAMES, self.parts)}
            self.stats['component_cosines'] = [[cosine(a, b) for b in self.parts] for a in self.parts]
            # Preserve the actual FP32 vectors for independent CPU reconstruction.
            rt.checkpoint(self.folder / f'vectors_u{self.u}.pt',
                          dict(names=self.names, sizes=[p.numel() for p in self.params],
                               gradient=self.gradient.cpu(), displacement=actual.cpu(),
                               weighted_components=self.parts.cpu()))
        with (self.folder / 'updates.jsonl').open('a') as f:
            f.write(json.dumps(self.stats, allow_nan=False) + '\n')
        del self.theta, self.gradient, self.parts


def build_instrumented_update(tracker):
    # Copy the frozen function verbatim, inserting read-only observation calls.
    # Save the exact executed source for review. No old file is modified.
    source = inspect.getsource(base.update)
    replacements = [
        ('    with b.step(u):', '    tracker.begin(m,o,u)\n    with b.step(u):'),
        ('                margin.backward(', '                tracker.feedback(margin,feedback,prob,B,i,inp["t"])\n                margin.backward('),
        ('        b.mark_optimizer_started();o.step();e.update(m,u);e.at_update_boundary=True',
         '        tracker.pre_step(m,o)\n        b.mark_optimizer_started();o.step();e.update(m,u);e.at_update_boundary=True'),
        ('    torch.cuda.synchronize()\n    return dict', '    tracker.finish(m,o)\n    torch.cuda.synchronize()\n    return dict')]
    for old, new in replacements:
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    namespace = dict(vars(base), tracker=tracker, check=check)
    exec(compile(source, str(HERE / 'instrumented_update.py'), 'exec'), namespace)
    (tracker.folder / 'executed_update.py').write_text(source)
    return namespace['update']


def main():
    idx = int(sys.argv[1]); protocol = read(OUT / 'protocol.json')
    task = protocol['tasks'][idx]; row = task['parent']
    folder = OUT / task['id']; folder.mkdir(exist_ok=False)
    started = time.time(); receipt = dict(status='running', start_epoch=started, updates=0)
    b = None
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(TimeoutError('supervisor stop')))
    try:
        rt.setup(); check()
        assert torch.cuda.is_available() and torch.cuda.device_count() == 1
        for path, sha in read(OUT / 'bindings.json').items():
            assert rt.sha(path) == sha, 'binding changed: ' + path
        z,cfg,case,m,o,e,rng,b,core,exact,canonical = base.setup(row)
        assert b.completed == 2000
        rt.save(folder / 'environment.json', g.common.environment('cuda'))
        rt.save(folder / 'identity.json', dict(task=task, config=cfg, optimizer_groups=[{k:v for k,v in x.items() if k!='params'} for x in o.param_groups]))
        history = [json.loads(x) for x in Path(row['history']).read_text().splitlines()]
        previous = {x['update']: x for x in map(json.loads, (PREV / task['id'] / 'training.jsonl').read_text().splitlines())}
        tracker = Tracker(m, folder); update = build_instrumented_update(tracker)
        base.check = check
        base.evaluation(folder,'u2000',z,cfg,case,m,o,e,rng,b,row)
        for u in range(2001,2051):
            check()
            if u in PROBES:
                rt.checkpoint(folder / f'boundary_u{u-1}.pt', dict(state=b.checkpoint(z['binding']), diagnostic_only=True))
            record = update(row,4,16,u,z,cfg,case,m,o,e,rng,b,core,exact,canonical,history)
            old = previous[u]
            for a,c in zip(record['records'],old['records']):
                for k in ('input_hash','query_hash','hard_uniform_hash','t'):
                    assert a[k] == c[k], 'historical input contract: ' + k
            assert record['main_rng_hash'] == old['main_rng_hash']
            record['historical_gradient_norm'] = old['gradient_norm']
            record['historical_max_risk_gap'] = max(abs(a-bb) for x,y in zip(record['records'],old['records']) for a,bb in zip(x['risks'],y['risks']))
            with (folder / 'training.jsonl').open('a') as f:
                f.write(json.dumps(record, allow_nan=False) + '\n')
            receipt['updates'] += 1
            rt.save(folder / 'progress.json',dict(completed=u, seconds=time.time()-started, last_seconds=record['seconds']))
            if u in (2012,2020,2050):
                base.evaluation(folder,f'u{u}',z,cfg,case,m,o,e,rng,b,row)
        state=b.checkpoint(z['binding']); rt.checkpoint(folder/'temporary_u2050.pt',dict(state=state,diagnostic_only=True))
        assert rt.sha(row['checkpoint'])==row['checkpoint_sha256']
        receipt.update(status='completed', extra_component_vjps=tracker.extra_vjps,
                       largest_component_sum_relative_error=max(tracker.sum_checks),
                       historical_inputs_and_RNG_match=True, CUDA_trajectory_bitwise_claim=False,
                       parent_unchanged=True)
    except BaseException as exc:
        receipt.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        if b is not None:
            try: rt.checkpoint(folder/'stopped_boundary.pt',dict(state=b.checkpoint(z['binding']),diagnostic_only=True))
            except BaseException as save_error: receipt['save_error']=repr(save_error)
    finally:
        receipt.update(end_epoch=time.time(), seconds=time.time()-started,
                       peak_GPU_MiB=torch.cuda.max_memory_allocated()/2**20 if torch.cuda.is_initialized() else 0)
        rt.save(folder/'receipt.json',receipt);print(json.dumps(receipt),flush=True)
    return 0 if receipt['status']=='completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
