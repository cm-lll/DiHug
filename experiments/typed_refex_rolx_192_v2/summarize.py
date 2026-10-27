"""Aggregate existing evidence, never run a model."""
import json,pathlib,sys,hashlib,time
HERE=pathlib.Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    evidence=pathlib.Path(sys.argv[1]);probe=evidence/'probe';summary=json.loads((probe/'summary.json').read_text());rows={}
    for label,row in summary['cases'].items():
        d={}
        for phase in ['cpu_repeat','same_device','cross_backend','restore']:
            raw=json.loads((probe/label/(phase+'.json')).read_text())
            modules={}
            for key,v in raw['gradients'].items():
                module=key.rsplit('.',1)[0];old=modules.setdefault(module,dict(max_absolute=0.,reference_max=0.,violations=0))
                old['max_absolute']=max(old['max_absolute'],v['max_abs']);old['reference_max']=max(old['reference_max'],v.get('reference_max',0.));old['violations']+=v['violations']
            d[phase]=dict(passed=raw['passes'],max_logit=max(x['logits']['max_abs'] for x in raw['inputs']),
                max_probability=max(x['p0']['max_abs'] for x in raw['inputs']),max_gradient=max(x['max_abs'] for x in raw['gradients'].values()),
                gradient_violations=sum(x['violations'] for x in raw['gradients'].values()),modules=modules,
                first_forward_hash_difference=raw.get('first_forward_hash_difference'))
            if phase=='restore':
                d[phase].update(parameters_passed=all(x['passes'] for x in raw['parameters'].values()),max_parameter=max(x['max_abs'] for x in raw['parameters'].values()),
                    optimizer_exact=raw['optimizer_exact'],rng_exact=raw['rng_exact'],max_optimizer_tensor=max(x['max_abs'] for x in raw['optimizer_tensors'].values()))
        dimensions=json.loads((probe/label/'dimensions.json').read_text())
        rows[label]=dict(numerical=d,parameter_count=dimensions['parameters'],load_exact=row['load_exact'],updates=row['actual_updates'],
            normal2_low_trace_seconds=row['normal2_first_seconds'],normal2_with_trace_seconds=row['normal2_traced_second_seconds'],full_query=row['full_query'],
            peak_allocated_mib=row['peak_allocated_mib'],peak_reserved_mib=row['peak_reserved_mib'],process_cumulative_peak_rss_mib=row['peak_rss_mib'],
            checkpoint_save_seconds=row['save_seconds'],feature_seconds_final_model_cumulative=row['feature_seconds'],encoding_seconds_final_model_cumulative=row['encoding_seconds'])
    old=HERE.parent/'typed_refex_rolx_192_v1/package';new=HERE/'package';core=[]
    for p in (new/'hsgd').rglob('*.py'):
        rel=p.relative_to(new);assert sha(p)==sha(old/rel);core.append(str(rel))
    for name in ['IMDB.npz','PubMed.npz']:assert sha(new/'assets'/name)==sha(old/'assets'/name)
    config=json.loads((new/'config.json').read_text());orig=json.loads((old/'config.json').read_text())
    for key in ['model','optimizer','precision','T','schedule_offset','tf32','seed','threads','tolerances']:assert config[key]==orig[key]
    result=dict(version='v2',scheme_consistent=True,formal_ready=False,formal_training=False,generated_graphs=0,
        executed_updates=sum(x['updates'] for x in rows.values()),engineering_tests={'passed':25,'failed':0,'skipped':0},
        cases=rows,unchanged_core_files=core,unchanged_data=True,prior_v1_preserved=True,
        synthetic_dictionary_limit='separate per-type rank4 synthetic NMF reached500 iterations in all eight fits; warnings preserved, no claim converged formal dictionary; NNLS KKT checked',
        conclusion='CPU repeats exact; GPU repeat forwards exact but initial gradients fail; load exact and resumed forward exact; resumed parameters within original tolerance, optimizer not exact => restore gate FAIL',
        missing=['numerical backward cause and remedy without relaxing gate','authorized formal dictionary fitting and convergence/cost','low-interference stable normal2 throughput beyond one update','actual forward-noise/time/state cost range','formal training updates per task and GPU allocation','12 full-chain/evaluation/IO costs'],
        three_hours_reliably_estimated=False)
    (evidence/'RESULTS.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:{x:v[x] for x in ['parameter_count','normal2_low_trace_seconds','peak_allocated_mib','checkpoint_save_seconds']} for k,v in rows.items()},indent=2))
if __name__=='__main__':main()
