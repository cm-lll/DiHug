"""Independent saved-record acceptance, without model calls or candidate sampling."""
import json
import hashlib
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/typed_edit_alignment_audit_20261118/cpu_v2'
read = lambda p: json.loads(Path(p).read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
sys.path.insert(0,str(ROOT/'research_branches/typed_edit_complexity_20261116'))
import common as b


def main():
    for path, digest in read(OUT/'manifest.json')['files'].items():
        assert sha(path) == digest, path
    rows = read(OUT/'stage1_records.json')
    graphs = {}
    candidate_checks = 0
    for r in rows:
        ds, seed, step = r['dataset'],r['start_seed'],r['step']
        key = ds,seed,step
        if key not in graphs:
            data = (b.RUN/'data/freebase' if ds=='freebase' else
                    ROOT/'results/typed_edit_revision_20261116/delicious/data')
            p = (data/f'starts/seed{seed}/initial.json' if step==0 else
                 ROOT/f'results/typed_edit_external_adaptation_20261117/{ds}_reachability_18000/outputs/M_{seed}/step{step}.json')
            g = b.restore_graph(read(p))
            graphs[key] = g,np.array(read(data/f'starts/seed{seed}/target.json'))
        g,target = graphs[key]
        batch = r['candidates']
        gains = []
        for i,f in enumerate(batch['family']):
            family = sorted(g.family_specs)[f]
            edges = {tuple(sorted(e)) for e in g.family_edges[family]}
            degrees = g.family_degrees(family).copy()
            old_error = sum((int(d)-int(t))**2 for d,t in zip(degrees,target[:,f]))
            for name,mask,sign in [('old_edges','old_valid',-1),('new_edges','new_valid',1)]:
                for edge,valid in zip(batch[name][i],batch[mask][i]):
                    if not valid:
                        continue
                    u,v = sorted(edge)
                    assert u != v
                    if sign == -1:
                        assert (u,v) in edges
                        edges.remove((u,v))
                    else:
                        assert (u,v) not in edges
                        edges.add((u,v))
                    degrees[u] += sign
                    degrees[v] += sign
            assert len(edges) == len(g.family_edges[family])
            fresh = np.zeros(g.num_nodes,dtype=np.int64)
            for u,v in edges:
                fresh[u] += 1
                fresh[v] += 1
            assert np.array_equal(fresh,degrees)
            gains.append(old_error-sum((int(d)-int(t))**2 for d,t in zip(fresh,target[:,f])))
            candidate_checks += 1
        assert np.array_equal(gains,r['degree_gains'])
        l = np.array(r['logits']); d = np.array(gains)
        assert np.argmax(l+r['effective_weight']*d/10) == r['selected']
        assert np.argmax(l+.05*d/10) == r['fixed_weight_selected']
        w = r['critical_weight_for_max_gain'] + 1e-5
        assert d[np.argmax(l+w*d/10)] == max(d)
    original = read(ROOT/'results/typed_edit_proposal_audit_20261117/records.json')
    selections = read(OUT/'stage2_selections.json')
    error = 0.
    for r in selections:
        pool = original[r['record']]['policies'][r['policy']]
        assert np.argmax(r['logits']) == r['selected']
        assert pool['triangle_deltas'][r['selected']] == r['triangle_delta']
        assert (pool['candidates']['action'][r['selected']]==0) == r['noop']
        if r['variant']=='original':
            old = pool['selections'][f'M_{r["seed"]}']
            error = max(error,float(np.max(np.abs(np.array(old['logits'])-r['logits']))))
            assert r['selected'] == old['index']
    assert error <= 2e-5
    summary = read(OUT/'summary.json')
    for group in summary['stage1']:
        rr = [r for r in rows if all(r[k]==group[k] for k in ['dataset','start_seed','step'])]
        assert len(rr)==group['batches']==8
        assert sum(r['selected_gain']<0 and r['maximum_gain']>0 for r in rr)==group['conflicts_with_positive']
        assert abs(np.mean([r['selected_gain'] for r in rr])-group['mean_selected_gain'])<1e-9
    for group in summary['stage2']:
        rr = [r for r in selections if all(r[k]==group[k] for k in ['state','policy','seed','variant'])]
        assert len(rr)==group['n']==32
        assert sum(r['noop'] for r in rr)==group['noop']
        assert sum(r['triangle_delta']<0 for r in rr)==group['negative']
    completion = read(OUT/'completion.json')
    assert len(rows)==112 and len(selections)==1536
    assert completion['scientific_model_forward_calls']==1648
    result = dict(passed=True,independent_edge_rebuilds=candidate_checks,
        stage1_batches=len(rows),stage2_selections=len(selections),original_logit_max_error=error,
        inputs_unchanged=True,model_calls_in_acceptance=0,
        output_hashes={p.name:sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='acceptance.json'},
        checker_sha256=sha(__file__))
    (OUT/'acceptance.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['output_hashes']},indent=2))


if __name__=='__main__':
    main()
