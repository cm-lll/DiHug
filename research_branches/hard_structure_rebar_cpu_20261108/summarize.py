"""Audit and summarize stored CPU estimator draws, never rerun science."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ[key]='1'
import csv,json,hashlib,sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/hard_structure_rebar_cpu_20261108'
rows=json.loads((OUT/'results.json').read_text())
panels=json.loads((OUT/'panels.json').read_text())
coordinates=json.loads((OUT/'selected_edges.json').read_text())
assets=json.loads((OUT/'asset_binding.json').read_text())

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def csvout(name,data):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in data for k in r)))
        w.writeheader();w.writerows(data)

summary=[];edge_rows=[]
for panel in panels:
    name=panel['panel'];by={r['method']:r for r in rows if r['panel']==name}
    raw,base,rebar,direct=[by[k] for k in ['raw_RF','baseline_RF','REBAR','edge_conditional']]
    with np.load(OUT/f'{name}_samples.npz') as z:
        for method in by:
            a=z[method]
            assert a.shape==(64,panel['selected_coordinates']) and np.isfinite(a).all()
            variance=float(a.var(0,ddof=1).sum())
            assert abs(variance-by[method]['coordinate_variance_sum'])<1e-15
            cc=[c for c in coordinates if c['panel']==name]
            for j,c in enumerate(cc):
                mean=float(a[:,j].mean());sd=float(a[:,j].std(ddof=1))
                edge_rows.append(dict(**c,method=method,mean_logit_gradient=mean,sd=sd,se=sd/8,
                    gradient_descent_probability_direction='increase' if mean<0 else 'decrease' if mean>0 else 'zero',
                    reference_label_opposed=(mean>0 if c['label']==1 else mean<0)))
    varratio=rebar['coordinate_variance_sum']/base['coordinate_variance_sum']
    costratio=rebar['evaluation_seconds']/base['evaluation_seconds']
    directratio=direct['coordinate_variance_sum']/base['coordinate_variance_sum']
    cost=direct['evaluation_seconds']/64
    # Deliberately labelled naive full-coordinate extrapolation, not benchmark.
    extrap=cost*panel['mutable_slots']/panel['selected_coordinates']
    summary.append(dict(panel=name,dataset=panel['dataset'],arm=panel['arm'],t=panel['t'],
        selected=panel['selected_coordinates'],eta=panel['eta'],
        rebar_vs_baseline_variance=varratio,rebar_vs_baseline_cost=costratio,
        rebar_vs_baseline_variance_cost=varratio*costratio,
        edge_conditional_vs_baseline_variance=directratio,
        baseline_seconds_per_sample=base['evaluation_seconds']/64,
        rebar_seconds_per_sample=rebar['evaluation_seconds']/64,
        selected_edge_conditional_seconds_per_sample=cost,
        naive_full_edge_conditional_seconds_per_sample=extrap,
        eta_development_seconds=panel['times']['development']['rebar_bundle_seconds'],
        rebar_paired_max_SE_units= rebar['paired_max_SE_units_vs_conditional']))

assert len(panels)==12 and len(rows)==48 and all(sha(p)==h for p,h in assets.items())
frozen=json.loads((OUT/'source_binding.json').read_text())
assert all(sha(p)==h for p,h in frozen.items())
manifest=json.loads((ROOT/'results/joint_output_formal_projected_20261108T043000/source_binding.json').read_text())
production_changes=[path for path,digest in manifest.items() if sha(path)!=digest]
assert not production_changes, 'Production source identity changed; report rather than alter it'
audit=dict(status='passed',panels=len(panels),method_rows=len(rows),selected_gradient_rows=len(edge_rows),
           input_files_unchanged=len(assets),execution_sources_unchanged=len(frozen),
           production_source_files_matching_frozen_manifest=len(manifest),
           rebar_lower_variance_than_baseline=sum(r['rebar_vs_baseline_variance']<1 for r in summary),
           rebar_lower_variance_cost_than_baseline=sum(r['rebar_vs_baseline_variance_cost']<1 for r in summary),
           all_mean_comparisons_descriptive=True,full_graph_gradient_variance_not_measured=True,
           GPU_hours=0,model_calls=0,formal_training_authorized=False)
csvout('comparison.csv',summary);csvout('per_edge_estimates.csv',edge_rows)
(OUT/'comparison.json').write_text(json.dumps(summary,indent=2))
(OUT/'audit.json').write_text(json.dumps(audit,indent=2))
print(json.dumps(audit))
for r in summary:
    print(r['panel'],'var',round(r['rebar_vs_baseline_variance'],4),
          'costvar',round(r['rebar_vs_baseline_variance_cost'],3),
          'direct',format(r['edge_conditional_vs_baseline_variance'],'.3g'),
          'ms',round(1000*r['rebar_seconds_per_sample'],2),
          'full_direct_seconds',round(r['naive_full_edge_conditional_seconds_per_sample'],2))
