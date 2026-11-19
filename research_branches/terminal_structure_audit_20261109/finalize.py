"""CPU-only aggregation of completed audit outputs; no new scientific calculations."""
import csv,hashlib,json,statistics,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/terminal_structure_audit_20261109'
CODE=ROOT/'research_branches/terminal_structure_audit_20261109'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    curves=json.loads((OUT/'cpu/curves.json').read_text());rows=[]
    keys=['terminal_cost','anonymous_joint_degree','typed_triangles','connected_probability','isomorphic_collision']
    for arm in sorted({x['arm'] for x in curves}):
        for step in [0,40]:
            sub=[x for x in curves if x['arm']==arm and x['step']==step]
            for k in keys:
                values=[r[k] for r in sub]
                rows.append(dict(arm=arm,step=step,metric=k,mean=statistics.mean(values),
                                 sample_sd=statistics.stdev(values),n_arithmetic_initializations=len(values),
                                 is_real_model_result=False))
    (OUT/'aggregate.json').write_text(json.dumps(rows,indent=2)+'\n')
    with (OUT/'aggregate.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
    bindings=json.loads((OUT/'cpu/binding.json').read_text())['assets']
    bindings.update(json.loads((OUT/'adapter/receipt.json').read_text())['hashes'])
    for p,h in bindings.items():assert sha(Path(p))==h,p
    sources=json.loads((OUT/'sources/manifest.json').read_text())
    for row in sources:
        if row['status']=='downloaded_not_executed':
            name='GDPO' if 'GDPO' in row['repository'] else 'GenStat'
            assert sha(OUT/'sources'/name/row['file'])==row['sha256']
    timing=[]
    for ds,seed in [('IMDB',2026110411),('IMDB',2026110412),('PubMed',2026110421),('PubMed',2026110422)]:
        d=ROOT/f'results/cn_multiseed_20261104T042546/run/{ds}_{seed}_A_late/ema_0.99'
        for p in sorted(d.glob('sample_*/summary.json')):
            z=json.loads(p.read_text())
            timing.append(dict(dataset=ds,train_seed=seed,sample_seed=z['sample_seed'],steps=z['steps'],
                               seconds=z['seconds'],file=str(p),sha256=sha(p),historical_not_current_throughput=True))
    (OUT/'historical_chain_cost.json').write_text(json.dumps(timing,indent=2)+'\n')
    receipts={name:json.loads((OUT/name/'receipt.json').read_text()) for name in ['cpu','adapter']}
    gaps=['real structural score contract not frozen for terminal fine-tuning',
          'real on-policy gradient variance and overhead not measured',
          'O/A/GPU random-context replay not validated by toy tests',
          'eager surrogate is biased; official GDPO normalization/clipping not reproduced',
          'toy composite risk improvement contains triangle costs; no production quality evidence']
    result=dict(status='complete_stopped',receipts=receipts,formal_model_calls=0,GPU_hours=0,
                formal_training_updates=0,real_generated_graphs=0,toy_parameter_updates=360,
                old_assets_unchanged=True,unresolved=gaps,external_code_execution=False,
                formal_training_started=False,
                suggested_next_bound=dict(wall_minutes=80,GPU_hours=2.5,max_GPU=2,
                    release_by_minute=65,final_CPU_minutes=15,
                    example_max_occupancy_GPU_hours=2*65/60,
                    new_full_chains=8,extra_forward_backward_timepoints=64,optimizer_updates=0,
                    status='not_started_requires_objective_and_asset_gate',
                    note='Historical extrapolation only; no promise of current throughput. Never hold both GPUs for full 80 minutes.'))
    (OUT/'completion.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    files=[p for p in OUT.rglob('*') if p.is_file() and 'sources' not in p.relative_to(OUT).parts]
    files.extend([OUT/'sources/manifest.json']+[p for p in CODE.glob('*') if p.is_file()])
    manifest={str(p.relative_to(ROOT)):sha(p) for p in files}
    (OUT/'delivery_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(OUT/'review.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,p.relative_to(ROOT))
    (OUT/'delivery.json').write_text(json.dumps(dict(finished_epoch=time.time(),
        review_sha256=sha(OUT/'review.zip'),source_hashes={str(p.relative_to(ROOT)):sha(p) for p in CODE.glob('*.py')},
        no_external_source_redistribution=True,no_weights_or_datasets=True),indent=2)+'\n')
    print(json.dumps(dict(completion=result,archive_bytes=(OUT/'review.zip').stat().st_size)))
if __name__=='__main__':main()
