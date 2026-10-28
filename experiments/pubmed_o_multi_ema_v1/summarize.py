"""Identity-based raw/EMA comparison using unchanged graph metric definitions."""
import argparse,pathlib,time,json
import numpy as np
from runtime_contract import ROOT,SEEDS,SAMPLES,READOUTS,DECAYS,jobs,cell,read,sha,save,source_binding,verify_readout
from summary_metrics import graph_summary,paired,strict_mean
from assessment import assess

def assemble(records,reference):
    template=graph_summary(reference,reference);cells={};contrasts=[];assessment_rows=[]
    index={}
    for row in records:
        k=(row['train_seed'],row['readout'],row['sample_seed'])
        if k in index:raise ValueError('duplicate identity')
        index[k]=row
    expected={(s,r,k) for s in SEEDS for r in READOUTS for k in SAMPLES}
    if set(index)!=expected:raise ValueError('must explicitly record all64 results, including missing ones')
    for seed in SEEDS:
        for r in READOUTS:
            rows=[index[seed,r,k] for k in SAMPLES];ready=all(x['status']=='complete' for x in rows)
            diversity=None
            if ready:
                a,b=[set(map(tuple,x['edge_set'])) for x in rows];union=len(a|b)
                diversity=dict(labelled_graphs_equal=a==b,edge_symmetric_difference=len(a^b),edge_jaccard=len(a&b)/union if union else 1.,scope='two labelled samples only; no topology/population diversity guarantee')
            cells[f'{cell(seed)}_{r}']=dict(graphs=rows,mean=strict_mean([x['values'] for x in rows]) if ready else None,
                                          mean_reason=None if ready else 'required_graph_missing_or_failed',diversity=diversity)
        for kind in DECAYS:
            diffs=[]
            for k in SAMPLES:
                left,right=index[seed,kind,k],index[seed,'raw',k]
                ready=left['status']==right['status']=='complete'
                diffs.append(dict(sample_seed=k,status='complete' if ready else 'incomplete',difference=paired(left.get('values'),right.get('values'),template)))
            contrasts.append(dict(train_seed=seed,contrast=kind+'-raw',per_sample=diffs,mean=strict_mean([x['difference'] for x in diffs]) if all(x['status']=='complete' for x in diffs) else None))
    for row in records:
        a={k:row[k] for k in ['train_seed','readout','sample_seed','status']}
        if row['status']=='complete':
            m=row['metrics'];a.update(edges=m['edges'],W1_incident4=m['W1_definitions']['incident4'],LCC=m['lcc'],family_edges=m['family_edges'],other_metrics=m)
        assessment_rows.append(a)
    return dict(status='complete' if all(x['status']=='complete' for x in records) else 'incomplete',cells=cells,contrasts=contrasts,
                assessment=assess(assessment_rows),scope='one trajectory per training seed; old2 and new6 separately; three EMA scales are repeated readouts, not independent trainings or cross-dataset evidence')

def collect(run,check=lambda:None):
    reference=read(ROOT/'reference.json');records=[]
    with np.load(ROOT/'package/assets/PubMed.npz',allow_pickle=False) as z:types=z['types'];schema=z['schema']
    for job in jobs()['sampling']:
        check();folder=run/job['directory'];row=dict(train_seed=job['train_seed'],readout=job['readout'],sample_seed=job['sample_seed'],status='incomplete',path=str(folder))
        try:
            if (folder/'summary.json').exists():row['worker_receipt']=read(folder/'summary.json')
            else:row['worker_receipt']=dict(status='not_started' if not folder.exists() else 'incomplete_no_receipt')
            if row['worker_receipt']['status']!='complete':raise ValueError('worker not complete')
            path,receipt=verify_readout(run,job);b=read(folder/'binding.json');m=read(folder/'metrics.json')
            if b['job']!=job or b['readout']!=job['readout'] or b['model_sha256']!=sha(path) or b['checkpoint_sha256']!=receipt['checkpoint_sha256']:raise ValueError('sample/readout identity mismatch')
            if b['source_sha256']!=source_binding()[1] or sha(folder/'graph.npz')!=b['graph_sha256'] or sha(folder/'metrics.json')!=b['metrics_sha256']:raise ValueError('sample/source hashes differ')
            if m['reference_metrics']!=reference:raise ValueError('reference differs')
            with np.load(folder/'graph.npz',allow_pickle=False) as z:
                np.testing.assert_array_equal(z['types'],types);np.testing.assert_array_equal(z['schema'],schema);edges=z['edges'];fam=z['families']
            if edges.shape!=(2,m['edges']) or fam.shape!=(m['edges'],) or np.any(edges<0) or np.any(edges>=len(types)) or np.any(edges[0]>=edges[1]) or len(set(map(tuple,edges.T)))!=m['edges']:raise ValueError('graph contract')
            if np.any(fam<0) or np.any(fam>=len(schema)):raise ValueError('family bounds')
            np.testing.assert_array_equal(np.sort(types[edges],axis=0).T,schema[fam])
            np.testing.assert_array_equal(np.bincount(fam,minlength=len(schema)),m['family_edges'])
            if not np.isclose(m['W1_definitions']['all10_compat'],.4*m['W1_definitions']['incident4'],atol=1e-12,rtol=1e-12):raise ValueError('W1 conventions differ')
            row.update(status='complete',values=graph_summary(m,reference),metrics=m,edge_set=edges.T.tolist(),binding=b)
        except (TimeoutError,InterruptedError):raise
        except Exception as e:row['reason']=repr(e)
        records.append(row)
    # Initial state/random stream pairing must match across readouts and training seeds.
    for k in SAMPLES:
        rows=[x for x in records if x['sample_seed']==k and x['status']=='complete']
        if rows:
            first=rows[0]['binding']
            for row in rows:
                if row['binding']['initial_state_sha256']!=first['initial_state_sha256'] or row['binding']['initial_rng']!=first['initial_rng']:
                    row.update(status='incomplete',reason='sampling initial state/RNG pairing differs');row.pop('values',None)
    return assemble(records,reference)

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);p.add_argument('--deadline',type=float,required=True);a=p.parse_args()
    run=pathlib.Path(a.run);out=pathlib.Path(a.output)
    if out.exists():raise FileExistsError('do not overwrite summary')
    def check():
        if time.time()>=a.deadline:raise TimeoutError('summary deadline')
    try:check();r=collect(run,check);check()
    except BaseException as e:r=dict(status='incomplete',summary_error=repr(e))
    save(out,r);return 0 if r['status']=='complete' else 2
if __name__=='__main__':raise SystemExit(main())
