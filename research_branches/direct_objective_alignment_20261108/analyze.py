"""Read-only CPU aggregation. No torch, model, RNG calls, or graph construction."""
import csv, hashlib, json, math, time, resource
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / 'results/direct_structure_response_20261108T205816'
OUT = ROOT / 'results/direct_objective_alignment_20261108'
OUT.mkdir(exist_ok=True)
START = time.time()
FILES = [OLD / n for n in ['conditional_graph_samples.csv', 'conditional_graph_paired.csv',
    'paired_metrics.csv', 'protocol.json', 'conditional_graph_audit.json']]
FILES += [ROOT/'research_branches/direct_structure_cpu_20261108/score.py',
          ROOT/'research_branches/direct_structure_response_20261108/conditional_graph_audit.py']
FILES += [OUT/'start_sample_metrics.csv',OUT/'start_reconstruction.json']
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
bindings = {str(p): sha(p) for p in FILES}
def read_csv(p):
    with p.open() as f: return list(csv.DictReader(f))
def save_csv(name, rows):
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
rows=read_csv(FILES[0]); old_pairs=read_csv(FILES[1]); probes=read_csv(FILES[2])
assert len(rows)==768
rows+=read_csv(OUT/'start_sample_metrics.csv')
assert len(rows)==1152
metrics=['DIRECT','type_equal_total_degree_W1','type_equal_relation_degree_W1',
         'type_equal_joint_degree_TV','triangle_abs_error','triangles','old_MMD_score']
long=[]; triangles=[]; panels=[]; grouped={}; max_residual=0.
for condition in sorted({r['condition'] for r in rows}):
    ds=condition.split('_')[0]; ref=45 if ds=='IMDB' else 1937
    degree_scale=2*(323 if ds=='IMDB' else 926)/192
    for t in [1,500,1000]:
        values={}
        for arm in ['start','CE','CE_STRUCT']:
            rr=sorted([r for r in rows if r['condition']==condition and int(r['t'])==t and r['arm']==arm],key=lambda r:int(r['sample']))
            assert [int(r['sample']) for r in rr]==list(range(32))
            values[arm]={k:np.array([float(r[k]) for r in rr]) for k in metrics}
            v=values[arm]
            reconstructed=(v['type_equal_total_degree_W1']/degree_scale+v['type_equal_joint_degree_TV']+v['triangle_abs_error']/ref)/3
            max_residual=max(max_residual,float(abs(reconstructed-v['DIRECT']).max()))
            assert np.allclose(reconstructed,v['DIRECT'],rtol=0,atol=1e-12)
            a=v['triangles']; excess=np.maximum(a-ref,0).mean(); deficit=np.maximum(ref-a,0).mean()
            bias=abs(a.mean()-ref); gap=v['triangle_abs_error'].mean()-bias
            assert abs(gap-2*min(excess,deficit))<1e-9
            p=next(r for r in probes if r['condition']==condition and int(r['t'])==t)
            suffix='start' if arm=='start' else ('CE' if arm=='CE' else 'STRUCT')
            exact=float(p['expected_triangles_'+suffix])
            triangles.append(dict(condition=condition,t=t,arm=arm,n=32,reference=ref,
                sample_mean=float(a.mean()),sample_sd=float(a.std(ddof=1)),
                exact_probability_triangle_mean=exact,sample_mean_minus_exact=float(a.mean()-exact),
                mean_absolute_error=float(v['triangle_abs_error'].mean()),absolute_sample_mean_error=float(bias),
                excess_mean=float(excess),deficit_mean=float(deficit),cross_reference_gap=float(gap),
                below_reference=int((a<ref).sum()),above_reference=int((a>ref).sum()),equal_reference=int((a==ref).sum())))
        a,b=values['CE'],values['CE_STRUCT'];grouped[condition,t]=values
        panel={'condition':condition,'t':t}
        old=next(r for r in old_pairs if r['condition']==condition and int(r['t'])==t)
        for k in metrics:
            d=b[k]-a[k];se=float(d.std(ddof=1)/math.sqrt(len(d)))
            assert abs(d.mean()-float(old[k+'_delta']))<1e-10
            assert abs(se-float(old[k+'_paired_MC_SE']))<1e-10
            long.append(dict(condition=condition,t=t,metric=k,n=32,CE_mean=float(a[k].mean()),
                candidate_mean=float(b[k].mean()),CE_sd=float(a[k].std(ddof=1)),candidate_sd=float(b[k].std(ddof=1)),
                paired_delta=float(d.mean()),paired_MC_SE=se,paired_lower=int((d<0).sum()),
                paired_equal=int((d==0).sum()),paired_higher=int((d>0).sum())))
            panel[k+'_delta']=float(d.mean());panel[k+'_MC_SE']=se
            panel[k+'_start']=float(values['start'][k].mean())
            for arm in ['CE','CE_STRUCT']:
                ds=values[arm][k]-values['start'][k]
                panel[k+'_'+arm+'_minus_start']=float(ds.mean())
                panel[k+'_'+arm+'_minus_start_MC_SE']=float(ds.std(ddof=1)/math.sqrt(len(ds)))
        for name,k,scale in [('degree','type_equal_total_degree_W1',degree_scale),('joint','type_equal_joint_degree_TV',1),('triangle','triangle_abs_error',ref)]:
            panel[name+'_score_contribution_delta']=float((b[k]-a[k]).mean()/scale/3)
        assert abs(sum(panel[k+'_score_contribution_delta'] for k in ['degree','joint','triangle'])-panel['DIRECT_delta'])<1e-12
        panels.append(panel)
summary=[]
for c in sorted({r['condition'] for r in rows}):
    s={'condition':c,'training_seed':2026110411 if c.startswith('IMDB') else 2026110422,
       'time_rule':'equal descriptive mean of t1,500,1000; NOT uniform-time risk'}
    for k in metrics:
        s[k+'_start']=float(np.mean([grouped[c,t]['start'][k].mean() for t in [1,500,1000]]))
        s[k+'_CE']=float(np.mean([grouped[c,t]['CE'][k].mean() for t in [1,500,1000]]))
        s[k+'_candidate']=float(np.mean([grouped[c,t]['CE_STRUCT'][k].mean() for t in [1,500,1000]]))
        s[k+'_delta']=s[k+'_candidate']-s[k+'_CE']
    summary.append(s)
save_csv('paired_metrics.csv',long);save_csv('triangle_decomposition.csv',triangles)
save_csv('panels.csv',panels);save_csv('three_time_summary.csv',summary)
save_csv('reused_sample_metrics.csv',rows)
assert bindings=={str(p):sha(p) for p in FILES}
out={'scope':'1152 existing raw conditional Bernoulli sample metric rows: 384 reconstructed starts plus 768 saved endpoints; 12 paired panels; no new draws/model/GPU/updates/full chains',
 'limits':['Two training seeds, each with two local histories, not four independent seeds.',
 '32 sample-pair MC SE describes fixed probability-panel sampling error, not training uncertainty or a significance test.',
 'Three viewed Xt states per parent do not estimate uniform-time training risk.',
 'Start draws reconstructed with registered original seeds and exact old-score checks; no new random panel.',
 'Cross-reference gap is a mean absolute error decomposition, NOT a general variance contribution.'],
 'sample_count':len(rows),'score_reconstruction_max_error':max_residual,'summary':summary,'panels':panels,
 'cpu_seconds':time.process_time(),'wall_seconds':time.time()-START,'max_rss_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
 'new_model_calls':0,'new_updates':0,'new_graph_samples':0,'GPU_hours':0,'assets_unchanged':True}
(OUT/'asset_binding.json').write_text(json.dumps(bindings,indent=2))
(OUT/'summary.json').write_text(json.dumps(out,indent=2))
(OUT/'source_binding.json').write_text(json.dumps({str(Path(__file__).resolve()):sha(Path(__file__))},indent=2))
print(json.dumps({'summary':summary,'triangles':triangles,'wall_seconds':out['wall_seconds'],'max_rss_MiB':out['max_rss_MiB']},indent=2))
