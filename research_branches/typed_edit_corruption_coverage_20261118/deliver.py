"""Read-only decomposition and delivery validation; no model forward or update."""
from train import OUT, SEEDS, b, data, np, put, verify_train
import ast
import csv
from pathlib import Path


def main():
    verify_train()
    accepted = b.read(OUT/'training_acceptance.json')
    summary = b.read(OUT/'summary.json')
    assert accepted['passed'] and not summary['short_generation_gate_passed']
    _, _, splits = data()
    z = splits['old_assessment']; y = z['y']; ix = np.arange(len(y))
    actions = z['x'][:, :, :5].argmax(-1)[ix, y]
    rows = []
    for seed in SEEDS:
        for kind in ['base', 'U', 'B']:
            saved = np.load(OUT/f'fits/{seed}/{kind}/old_assessment.npz')
            assert np.array_equal(saved['targets'], y)
            q = saved['logits'].astype(float); q -= q.max(1, keepdims=True)
            ce = np.log(np.exp(q).sum(1)) - q[ix, y]
            for action, name in enumerate(['noop', 'birth', 'death', 'relocation', 'switch']):
                mask = actions == action
                rows.append(dict(seed=seed, kind=kind, action=name, n=int(mask.sum()),
                    full_correct=int((q.argmax(1)==y)[mask].sum()),
                    full_accuracy=float((q.argmax(1)==y)[mask].mean()),
                    full_ce=float(ce[mask].mean()), ce_contribution=float(ce[mask].sum()/len(y))))
            assert abs(sum(r['ce_contribution'] for r in rows if r['seed']==seed and r['kind']==kind)-ce.mean()) < 1e-12
    with (OUT/'posthoc_action_metrics.csv').open('w') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    agg=[]
    for kind in ['base','U','B']:
        for action in ['noop','birth','death','relocation','switch']:
            r=[v for v in rows if v['kind']==kind and v['action']==action]
            row=dict(kind=kind, action=action, n_per_seed=r[0]['n'], existing_parent_seeds=2)
            for k in ['full_accuracy','full_ce','ce_contribution']:
                row[k+'_mean']=float(np.mean([v[k] for v in r]))
                row[k+'_sample_sd']=float(np.std([v[k] for v in r],ddof=1))
            agg.append(row)
    with (OUT/'posthoc_action_mean_sd.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(agg[0]));w.writeheader();w.writerows(agg)
    for p,h in b.read(OUT/'cache_acceptance.json')['output_hashes'].items():
        assert b.sha(p)==h,p
    source=Path(__file__).parent
    for p in source.glob('*.py'): ast.parse(p.read_text())
    for name in ['REPORT_ZH.md','REVIEW_RESPONSE_ADDENDUM_EN.md','per_seed_metrics.csv','mean_sd.csv']:
        assert (OUT/name).stat().st_size > 0
    endpoints=b.read(OUT/'cache_acceptance.json')['rows']
    put('delivery.json',dict(completed=True,scientific_gate_passed=False,candidate_adopted=False,
        connected_unique_corruption_endpoints=sum(r['connected'] for r in endpoints),
        disconnected_corruption_endpoints=[dict(kind=r['kind'],trajectory_id=r['trajectory_id']) for r in endpoints if not r['connected']],
        intermediate_connectivity_exhaustively_checked=False,
        scientific_training_updates=800,engineering_training_updates=34,verification_replay_updates=800,
        unique_corruption_trajectories=18,unique_cache_records=576,logical_cache_records=768,
        scientific_corruption_steps=2304,new_independent_from_scratch_fits=0,
        new_surrogate_generation_chains=0,gpu_calls=0,
        posthoc_diagnostic='saved-logit action decomposition only; no new model calls',
        all_prior_manifest_hashes_verified=True,
        source_hashes={str(p):b.sha(p) for p in source.iterdir() if p.is_file()},
        result_hashes={str(p):b.sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='delivery.json'},
        next_decision='Consider a separately frozen replay-composition test retaining noop and both switch directions; no automatic training or generation.'))
    print('Delivery validated; generation gate failed; zero surrogate chains.')


if __name__=='__main__': main()
