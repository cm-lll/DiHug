"""Replay only saved chosen edits with Python edge sets and integer degree ledgers."""
import json
from pathlib import Path
import numpy as np
import audit as a
import metrics

b=a.b
OUT=a.ROOT/'results/typed_edit_alignment_audit_20261118/progress_filter_pilot_128'


def main():
    for path,h in b.read(OUT/'manifest.json')['files'].items():
        assert b.sha(path)==h,path
    outputs=b.read(OUT/'per_output.json')
    assert len(outputs)==12
    checks=[]
    references={}
    first={}
    for row in outputs:
        ds,seed,arm=row['dataset'],row['start_seed'],row['arm']
        data,_,folder=a.spec(ds,seed)
        graph=b.restore_graph(b.read(folder/'step10000.json'))
        target=np.array(b.read(data/f'starts/seed{seed}/target.json'))
        names=sorted(graph.family_specs)
        edges={f:{tuple(sorted(e)) for e in graph.family_edges[f]} for f in names}
        degrees=np.column_stack([graph.family_degrees(f) for f in names]).astype(np.int64)
        record=b.read(OUT/'outputs'/row['output_key']/'trace.json')
        previous=int(np.square(degrees-target).sum())
        first.setdefault((ds,seed),[]).append(record['moves'][0]['candidates_sha256'])
        for step,r in enumerate(record['moves'],1):
            assert r['step']==step
            gain=np.array(r['gains']);logits=np.array(r['logits']);score=np.array(r['scores'])
            allowed=np.ones(len(gain),dtype=bool)
            if arm!='original_M':
                if np.any(gain>0):
                    allowed=gain>0
                else:
                    # Original production construction always prepends the unique noop.
                    allowed=np.zeros(len(gain),dtype=bool);allowed[0]=True
            assert np.array_equal(allowed,r['allowed'])
            assert np.argmax(np.where(allowed,score,-np.inf))==r['selected']
            if arm!='filtered_N':
                expected=logits+record['effective_weight']*gain/10
                assert np.max(abs(expected-score))<1e-4
            else:
                assert np.all(logits==0)
                assert gain[r['selected']]==max(gain)
            chosen=r['candidate'];f=names[chosen['family']]
            for key,valid,sign in [('old_edges','old_valid',-1),('new_edges','new_valid',1)]:
                for edge,yes in zip(chosen[key],chosen[valid]):
                    if not yes:continue
                    u,v=sorted(edge);assert u!=v
                    if sign<0:
                        assert (u,v) in edges[f];edges[f].remove((u,v))
                    else:
                        assert (u,v) not in edges[f];edges[f].add((u,v))
                    degrees[u,chosen['family']]+=sign
                    degrees[v,chosen['family']]+=sign
            current=int(np.square(degrees-target).sum())
            assert previous-current==r['selected_gain']==gain[r['selected']]
            assert current/graph.num_nodes==r['error_per_node']
            if arm!='original_M':assert current<=previous
            previous=current
        final_path=OUT/'outputs'/row['output_key']/'graph.json'
        assert b.sha(final_path)==row['graph_sha256']
        final=b.restore_graph(b.read(final_path));final.validate()
        for f in names:
            assert edges[f]=={tuple(sorted(e)) for e in final.family_edges[f]}
            assert len(edges[f])==len(graph.family_edges[f])
        assert np.array_equal(np.column_stack([final.family_degrees(f) for f in names]),degrees)
        if ds not in references:
            ref=b.restore_graph(b.read(data/'graph.json'));references[ds]=ref,metrics.counts(ref)
        ref,counts=references[ds]
        evaluated=metrics.compare(final,ref,counts)
        for key in ['primary_error','typed_triangle_tv','connected','lcc_fraction','typed_triangles','cross_relation_four_cycles']:
            assert evaluated[key]==row[key],key
        assert row['steps']==len(record['moves'])<=128
        assert row['arrived']==(row['final_error_per_node']<=5.5)
        if arm=='filtered_N':assert record['model_calls']==0
        checks.append(dict(output=row['output_key'],steps=len(record['moves']),passed=True))
    assert all(len(set(x))==1 for x in first.values())
    decision=b.read(OUT/'decision.json')
    for c in decision['checks']:
        rows={r['arm']:r for r in outputs if r['dataset']==c['dataset'] and r['start_seed']==c['start_seed']}
        m,n=rows['filtered_M'],rows['filtered_N']
        assert c['learned_beats_matched_N']==(m['primary_error']<n['primary_error'])
        assert c['structure_guard']==(m['primary_error']<=m['initial_metrics']['primary_error']*1.05)
    result=dict(passed=True,outputs=checks,independent_saved_edit_replays=sum(r['steps'] for r in checks),
        original_inputs_unchanged=True,model_calls=0,checker_sha256=b.sha(__file__),
        decision_sha256=b.sha(OUT/'decision.json'))
    (OUT/'acceptance.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='outputs'},indent=2))


if __name__=='__main__':main()
