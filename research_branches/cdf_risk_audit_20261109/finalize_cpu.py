import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['CUDA_VISIBLE_DEVICES']=''
import csv,hashlib,json,time,resource,importlib.util
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/cdf_risk_audit_20261109';RAW=OUT/'attempt2';START=time.monotonic()
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,o):
    (OUT/(name+'.json')).write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    if isinstance(o,list) and o:
        with (OUT/(name+'.csv')).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in o for k in r)));w.writeheader();w.writerows(o)
binding=read(RAW/'binding.json')
for p,s in binding.items():
    if Path(p).name=='run_cpu.py':assert sha(RAW/'source_at_execution.py')==s
    else:assert sha(p)==s
registrations={};identities=[]
for b in read(RAW/'probability_binding.json'):
    task=b['task']
    if task not in registrations:
        rp=ROOT/'results/structure_input_formal_20261106T045833/run'/task/'sampling_registration.json';r=read(rp)
        assert r['configuration']=='STRUCTURE_late' and sha(r['readout_file'])==r['readout_sha256']
        registrations[task]=r
        identities.append(dict(task=task,actual_configuration=r['configuration'],architecture='A late CN plus node triangle and component inputs, pair same-component input',readout='EMA0.99',updates=3000,readout_file=r['readout_file'],readout_sha256=r['readout_sha256'],registration_sha256=sha(rp)))
    assert registrations[task]['readout_sha256']==b['readout_sha256']
save('corrected_probability_identity',identities)
for name in ['conditional_summary','conditional_groups','directional_derivatives','selected_edge_derivatives']:
    rows=read(RAW/(name+'.json'))
    for r in rows:r.update(model_family='STRUCTURE_late (not ordinary A/PB/JK/PS)',readout='EMA0.99',updates=3000)
    save(name,rows)
cond=read(OUT/'conditional_summary.json');group=read(OUT/'conditional_groups.json');dirs=read(OUT/'directional_derivatives.json');edges=read(OUT/'selected_edge_derivatives.json')
agg=[]
for ds in ['IMDB','PubMed']:
    for kind in ['forward','generated']:
        rr=[r for r in cond if r['dataset']==ds and r['kind']==kind]
        b=float(np.mean([r['bias'] for r in rr]));v=float(np.mean([r['variance'] for r in rr]));agg.append(dict(dataset=ds,kind=kind,states=len(rr),mean_bias=b,mean_variance=v,mean_risk=b+v,variance_fraction=v/(b+v),variance_to_bias=v/b))
save('conditional_aggregate',agg)
# Independent reconstruction of every finite-four-output scalar and CDF identity.
graphs=read(RAW/'per_graph.json');summaries=read(RAW/'graph_CDF_decomposition.json')
for r in summaries:
    rr=[x for x in graphs if all(x[k]==r[k] for k in ['panel','dataset','train_seed','arm'])]
    assert len(rr)==4 and abs(np.mean([x['PB'] for x in rr])-r['mean_graph_PB'])<1e-12
    assert abs(r['mean_graph_PB']-r['pooled_bias']-r['sample_dispersion'])<1e-12
scalars=read(RAW/'scalar_decomposition.json')
for r in scalars:
    x=np.array(r['values']);ref=r['reference']
    assert abs(np.mean((x-ref)**2)-r['bias_squared']-r['variance_population'])<1e-6*max(1,r['MSE'])
    assert abs(x.std(ddof=1)-r['sample_SD'])<1e-9
# Independent full exclusion-DP checks on real same-type groups; no model execution.
spec=importlib.util.spec_from_file_location('audit',ROOT/'research_branches/cdf_risk_audit_20261109/run_cpu.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
checks=[]
for b in read(RAW/'probability_binding.json'):
    z=m.load(b['probability_file']);ds=b['task'].split('_')[0];types,schema,ref=m.asset(ds);gs=m.make_groups(types,schema,ref,z['query'],z['family']);p=z['probabilities']
    for g in gs:
        if 'shared' not in g:continue
        v=p[g['indices'][0]];j=int(np.argmin(abs(v-.5)));h=m.pmf(v[None]);fast=m.remove_factor(h,np.array([v[j]]));direct=m.pmf(np.delete(v,j)[None]);error=float(np.max(abs(fast-direct)));assert error<2e-11
        checks.append(dict(task=b['task'],kind=b['kind'],t=b['t'],relation=g['relation'],max_error=error))
save('independent_exclusion_checks',checks)
save('audit',dict(status='complete',model_id_correction='Task suffix A_late did not establish actual architecture; registered STRUCTURE_late verified before final report. Numeric calculations unchanged; original attempt2 protocol and results preserved.',initial_failure='Missing metric_dir for newly computed PS rows, before real probability computation; resolved via actual evaluation/task/u3000/sample path and existing binding checks; no records omitted.',original_source_sha256=sha(RAW/'source_at_execution.py'),current_reproducer_sha256=sha(ROOT/'research_branches/cdf_risk_audit_20261109/run_cpu.py'),finite_distribution_tests=len(read(RAW/'tests.json')),real_exclusion_checks=len(checks),graph_models=len(summaries),graph_count=len(graphs),conditional_states=len(cond),selected_probability_derivatives=len(edges),source_files_verified=len(binding),GPU_hours=0,model_calls=0,updates=0,new_graphs=0,finalizer_seconds=time.monotonic()-START,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024))
print(json.dumps(read(OUT/'audit.json'),ensure_ascii=False))
