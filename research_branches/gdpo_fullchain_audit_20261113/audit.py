"""Read-only full-chain audit: scalar logs, saved graphs and CPU state tensors.

No denoiser construction, forward/backward, optimizer update, or graph generation.
Synthetic algebra checks are independent of the external implementation.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import csv
import hashlib
import json
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PANEL = ROOT / 'results/paper_ablation_20261112T190000'
OUT = ROOT / 'results/gdpo_fullchain_audit_20261113'


def load(path):
    return json.loads(Path(path).read_text())


def table(name, rows):
    if not rows:
        return
    with (OUT / name).open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def batch_directory(job, u):
    ds = job['dataset']
    if not job['cached_complete']:
        return PANEL / job['id'] / f'{ds}_T_eager'
    if u <= 2:
        return ROOT / f'results/three_arm_learning_pilot_20261111T211641/{ds}_T_eager'
    return ROOT / f'results/terminal_continuation_20261112T003000/{ds}_T_eager'


def parameter_group(name):
    if name.startswith('common_neighbor.'):
        return 'CN'
    if name.startswith('joint_head.'):
        return 'joint_head'
    if name.startswith('branch.'):
        return 'degree'
    if '.self_attn.' in name:
        layer = int(name.split('tf_layers.')[1].split('.')[0])
        return 'late_attention' if layer >= 4 else 'early_attention'
    return 'other_backbone_or_buffer'


def synthetic_checks():
    # Each array entry is one legal slot's CE; duplication changes sum, not mean.
    ce = np.array([.1, .3, 1.2])
    assert np.isclose(ce.mean(), np.tile(ce, 4).mean())
    assert np.isclose(np.tile(ce, 4).sum(), 4 * ce.sum())
    # d(C * log p)/d logit = C*(label-p). A negative advantage reinforces label.
    p, label, advantage = .3, 1., -.2
    derivative = advantage * (label - p)
    assert derivative < 0
    # Two-sample population z-score erases the magnitude of every nonzero gap.
    zscores = []
    for gap in (.001, .1, 1.):
        costs = np.array([.2, .2 + gap])
        z = (costs - costs.mean()) / costs.std()
        np.testing.assert_allclose(z, [-1., 1.], atol=1e-12)
        zscores.append(z.tolist())
    # First Adam step for a fixed positive scalar multiplier need not grow with it.
    g = np.array([1e-4, -1e-3, .2])
    step = lambda x: -1e-5 * x / (np.abs(x) + 1e-8)
    ratio = np.linalg.norm(step(g * 18336)) / np.linalg.norm(step(g))
    return dict(slot_duplicate_mean_invariant=True, sum_multiplies_by_four=True,
                cost_sign_correct=True, two_sample_zscores=zscores,
                first_Adam_norm_ratio_for_gradient_scale_18336=float(ratio),
                interpretation='Algebra fixtures, not a model-quality or optimizer-trajectory test.')


def main():
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = load(PANEL / 'protocol.json')
    refs = load(PANEL / 'reference_metrics.json')
    batches, relations, missing = [], [], []
    for job in protocol['jobs']:
        ds = job['dataset']
        target = np.array(refs[ds]['family_edges'], dtype=float)
        for u in range(1, 5):
            directory = batch_directory(job, u)
            path = directory / f'round{u}_batch.json'
            if not path.exists():
                missing.append(str(path)); continue
            data = load(path)
            terms = []
            for k, plan in enumerate(job['terminal_plan'][str(u)]):
                graph = directory / f'train_round{u}_seed{plan["seed"]}/graph.npz'
                with np.load(graph) as z:
                    counts = np.bincount(z['families'], minlength=len(target)).astype(float)
                per = ((counts - target) / np.maximum(1., target)) ** 2 / len(target)
                count_cost = float(per.sum()) if job['reward'] == 'typed' else float(
                    ((counts.sum() - target.sum()) / target.sum()) ** 2)
                joint = data['costs'][k] - count_cost
                assert joint > -1e-8, (job['id'], u, k, joint)
                terms.append((count_cost, joint, per))
            gap = float(data['costs'][0] - data['costs'][1])
            count_gap = terms[0][0] - terms[1][0]
            joint_gap = terms[0][1] - terms[1][1]
            np.testing.assert_allclose(gap, count_gap + joint_gap, atol=1e-12)
            batches.append(dict(job=job['id'], dataset=ds, reward=job['reward'], update=u,
                                cost0=data['costs'][0], cost1=data['costs'][1], gap=gap,
                                count_gap=count_gap, joint_gap=joint_gap,
                                count_joint_disagree=bool(count_gap * joint_gap < 0),
                                count_controls_ranking=bool(abs(count_gap) > abs(joint_gap)),
                                source=str(path.relative_to(ROOT))))
            if job['reward'] == 'typed':
                for family in range(len(target)):
                    relations.append(dict(job=job['id'], update=u, family=family,
                                          reference_edges=int(target[family]),
                                          cost0=float(terms[0][2][family]),
                                          cost1=float(terms[1][2][family]),
                                          contribution_to_pair_gap=float(terms[0][2][family] - terms[1][2][family])))
    table('batch_cost_decomposition.csv', batches)
    table('relation_count_contributions.csv', relations)

    import torch
    torch.set_num_threads(1)
    changes = []
    for job in protocol['jobs']:
        if job['reward'] != 'typed' or job['architecture'] != 'A':
            continue
        ds = job['dataset']
        directory = PANEL / job['id'] / f'{ds}_T_eager'
        first = directory / 'initial_complete.pt'
        if job['cached_complete']:
            first = batch_directory(job, 1) / 'initial_complete.pt'
        last = directory / 'final_complete.pt'
        old = torch.load(first, map_location='cpu', weights_only=False)['model']
        new = torch.load(last, map_location='cpu', weights_only=False)['model']
        totals = {}
        for name, before in old.items():
            if not torch.is_floating_point(before):
                assert torch.equal(before, new[name]); continue
            group = parameter_group(name)
            t = totals.setdefault(group, dict(numel=0, changed=0, delta2=0., weight2=0., max_abs=0.))
            delta = new[name].double() - before.double()
            t['numel'] += before.numel()
            t['changed'] += int(torch.count_nonzero(delta))
            t['delta2'] += float((delta * delta).sum())
            t['weight2'] += float(before.double().square().sum())
            t['max_abs'] = max(t['max_abs'], float(delta.abs().max()))
        total_delta2 = sum(v['delta2'] for v in totals.values())
        for group, v in totals.items():
            changes.append(dict(job=job['id'], group=group, numel=v['numel'],
                                changed=v['changed'], delta_norm=v['delta2'] ** .5,
                                relative_norm=(v['delta2'] / max(v['weight2'], 1e-300)) ** .5,
                                squared_displacement_share=v['delta2']/max(total_delta2, 1e-300),
                                max_abs=v['max_abs']))
        del old, new
    table('saved_parameter_changes.csv', changes)
    # Verify binding of the actual local science files examined; never repair in place.
    paths = ['research_branches/paper_ablation_20261112/worker.py',
             'research_branches/paper_ablation_20261112/scoring.py',
             'research_branches/one_step_vs_terminal_plan_20261111/gpu_preflight.py']
    bindings = {}
    for rel in paths:
        path = ROOT / rel
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        expected = protocol['bindings'][str(path)]
        bindings[rel] = dict(sha256=actual, matches_run=actual == expected)
        assert actual == expected
    result = dict(status='complete', seconds=time.time()-start, batch_records=len(batches),
                  parameter_groups=len(changes), missing=missing, bindings=bindings,
                  checks=synthetic_checks(), GPU_hours=0, model_calls=0,
                  training_updates=0, new_graphs=0,
                  limitations=['No causal intervention or model gradient computed.',
                               'Parameter deltas are actual saved-state differences, not attribution.',
                               'Training batches use different random trajectories.'])
    (OUT / 'audit.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
