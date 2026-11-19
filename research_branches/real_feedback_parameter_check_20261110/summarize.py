"""CPU summaries of frozen VJP Gram matrices; no model calls."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,csv,itertools
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/real_feedback_parameter_check_20261110'
NAMES=['relation_count','joint_degree','present_triangles','absent_triangles','components','mean']
def write(name,rows):
    if not rows:return
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def coeff(n,r,j):
    a=np.zeros(n)
    if j<5:a[r*5+j]=1
    else:a[r*5:r*5+5]=.2
    return a
def cosine(H,a,b):
    aa=max(0.,float(a@H@a));bb=max(0.,float(b@H@b));return float(a@H@b/np.sqrt(aa*bb)) if aa*bb>0 else None
def main():
    cases=json.loads((OUT/'conditions.json').read_text());pairs=[];groups=[];exact=[];coupling=[];modules=[];summ=[]
    for x in cases:
        meta={k:x[k] for k in ('dataset','seed','t')};z=np.load(OUT/f'{x["dataset"]}_{x["seed"]}_t{x["t"]}_grams.npz');P=z['parameter'];L=z['margin'];ref=np.zeros(len(P));ref[20]=1
        for j,name in enumerate(NAMES):
            for r,s in itertools.combinations(range(4),2):
                pairs.append(dict(**meta,group=name,repeat_a=r,repeat_b=s,margin_cosine=cosine(L,coeff(len(L),r,j),coeff(len(L),s,j)),parameter_cosine=cosine(P,coeff(len(P),r,j),coeff(len(P),s,j))))
        for r in range(4):
            pn=np.sqrt(np.maximum(np.diag(P)[r*5:r*5+5],0));ln=np.sqrt(np.maximum(np.diag(L)[r*5:r*5+5],0))
            for j,name in enumerate(NAMES[:5]):
                groups.append(dict(**meta,repeat=r,group=name,parameter_norm=float(pn[j]),margin_norm=float(ln[j]),parameter_norm_share=float(pn[j]/pn.sum()) if pn.sum() else 0.,margin_norm_share=float(ln[j]/ln.sum()) if ln.sum() else 0.))
                for k in range(j+1,5):coupling.append(dict(**meta,repeat=r,group_a=name,group_b=NAMES[k],parameter_cosine=cosine(P,coeff(len(P),r,j),coeff(len(P),r,k))))
            a=coeff(len(P),r,0);den=float(ref@P@ref)
            exact.append(dict(**meta,repeat=r,parameter_count_cosine=cosine(P,a,ref),relative_count_error=float(np.sqrt(max(0.,(a-ref)@P@(a-ref))/den)) if den>0 else None))
            a=coeff(len(P),r,5);total=float(a@P@a)
            for key in z.files:
                if key in ('margin','parameter'):continue
                M=z[key];modules.append(dict(**meta,repeat=r,module=key,mean_gradient_squared_norm_share=float(a@M@a/total) if total>0 else 0.,mean_gradient_norm=float(np.sqrt(max(0.,a@M@a)))))
    for ds in ('IMDB','PubMed'):
        for name in NAMES:
            a=[x for x in pairs if x['dataset']==ds and x['group']==name];pp=[x['parameter_cosine'] for x in a if x['parameter_cosine'] is not None];ll=[x['margin_cosine'] for x in a if x['margin_cosine'] is not None]
            g=[x for x in groups if x['dataset']==ds and x['group']==name]
            summ.append(dict(dataset=ds,group=name,nonzero_pairs=len(pp),margin_cosine_mean=float(np.mean(ll)) if ll else None,parameter_cosine_mean=float(np.mean(pp)) if pp else None,negative_parameter_pairs=sum(v<0 for v in pp),parameter_norm_share=float(np.mean([v['parameter_norm_share'] for v in g])) if g else None,zero_parameter_batches=sum(v['parameter_norm']==0 for v in g)))
    case_summary=[]
    for x in cases:
        a=[v for v in pairs if all(v[k]==x[k] for k in ('dataset','seed','t')) and v['group']=='mean'];e=[v for v in exact if all(v[k]==x[k] for k in ('dataset','seed','t'))]
        case_summary.append(dict(dataset=x['dataset'],seed=x['seed'],t=x['t'],margin_cosine=float(np.mean([v['margin_cosine'] for v in a])),parameter_cosine=float(np.mean([v['parameter_cosine'] for v in a])),min_parameter_cosine=float(min(v['parameter_cosine'] for v in a)),count_exact_parameter_cosine=float(np.mean([v['parameter_count_cosine'] for v in e])),seconds=x['condition_seconds']))
    for name,a in [('repeat_pairs.csv',pairs),('gradient_groups.csv',groups),('count_exact_comparison.csv',exact),('component_coupling.csv',coupling),('module_contributions.csv',modules),('summary.csv',summ),('case_summary.csv',case_summary)]:write(name,a)
    obj=dict(summary=summ,cases=case_summary,conditions_completed=len(cases),max_cache_probability_difference=max(x['cache_probability_max_difference'] for x in cases),max_linearity_relative_error=max(x['linearity_relative_error'] for x in cases),max_fresh_forward_relative_difference=max(x['fresh_forward_VJP_relative_difference'] for x in cases if x['fresh_forward_VJP_relative_difference'] is not None),all_states_unchanged=all(x['complete_state_unchanged'] for x in cases))
    (OUT/'summary.json').write_text(json.dumps(obj,indent=2,allow_nan=False));print(json.dumps(obj,indent=2))
if __name__=='__main__':main()
