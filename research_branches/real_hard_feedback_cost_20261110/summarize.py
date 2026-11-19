"""Read-only analysis of frozen panel. Analytic reference applies ONLY to counts."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,csv,time,hashlib,itertools,platform
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/real_hard_feedback_cost_20261110'
def read(n):return list(csv.DictReader((OUT/n).open()))
def write(n,rows):
    with (OUT/n).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for x in iter(lambda:f.read(1024*1024),b''):h.update(x)
    return h.hexdigest()
def cosine(a,b):
    d=float(np.linalg.norm(a)*np.linalg.norm(b));return float(a@b/d) if d else None
def main():
    start=time.time();assert json.loads((OUT/'receipt.json').read_text())['status']=='completed'
    errors=[]
    for p in (np.array([.1,.3,.8]),np.array([0.,1.,.5])):
        for m in range(4):
            b=np.array(list(itertools.product([0,1],repeat=3)));prob=np.prod(np.where(b,p,1-p),axis=1);cost=(b.sum(1)-m)**2/max(1,m)**2
            exact=(prob*cost)@(b-p);formula=p*(1-p)*(1-2*p+2*(p.sum()-m))/max(1,m)**2
            np.testing.assert_allclose(exact,formula,atol=1e-14,rtol=1e-12);errors.append(float(abs(exact-formula).max()))
    (OUT/'analytic_algebra_audit.json').write_text(json.dumps(dict(passed=True,exhaustive_3_edge_cases=8,max_error=max(errors),includes_boundary_probabilities=True),indent=2))
    batches=read('batches.csv');groups=read('groups.csv');pairs=read('repeat_pairs.csv');assets=json.loads((OUT/'assets.json').read_text())
    summary=[];component=[];analytic=[];sourceaudit=[];casedetail=[]
    # No new sampling; independently reconstruct existing draws and compare count
    # score gradients to their Bernoulli analytic expectation, not to a model Jacobian.
    for a in assets:
        folder=Path(a['prediction']).parent;ev=torch.load(folder/'evaluation_inputs.pt',map_location='cpu',weights_only=False)
        inp=next(x for x in ev['inputs'] if x['t']==a['t']);order=np.argsort(inp['ids']);fam=np.asarray(inp['f'])[order];y=np.asarray(inp['labels'])[order]
        z=np.load(OUT/f'{a["dataset"]}_{a["seed"]}_t{a["t"]}.npz');p=z['probability'];R=10;exact=np.zeros_like(p);expected=0.
        for f in range(R):
            ix=fam==f;mu=float(p[ix].sum());m=int(y[ix].sum());den=max(1,m)**2
            exact[ix]=p[ix]*(1-p[ix])*(1-2*p[ix]+2*(mu-m))/(den*R)
            expected+=(float(np.sum(p[ix]*(1-p[ix])))+(mu-m)**2)/(den*R)
        for rep in range(4):
            rng=np.random.default_rng(np.random.SeedSequence([2026111057,a['seed'],a['t'],rep]));b=(rng.random((64,len(p)))<p).astype(np.uint8)
            np.testing.assert_array_equal(b,z['bits'][rep]);cc=np.zeros(64)
            for f in range(R):cc+=((b[:,fam==f].sum(1,dtype=np.int64)-int(y[fam==f].sum()))/max(1,int(y[fam==f].sum())))**2/R
            np.testing.assert_allclose(cc,z['costs'][rep,:,0],rtol=1e-13,atol=1e-13)
            g=z['logit_gradients'][rep,0];n=float(np.linalg.norm(exact))
            analytic.append(dict(dataset=a['dataset'],seed=a['seed'],t=a['t'],repeat=str(rep),count_expected_cost=expected,sampled_cost_mean=float(cc.mean()),exact_logit_gradient_norm=n,estimated_norm=float(np.linalg.norm(g)),cosine_to_exact=cosine(g,exact),relative_gradient_error=float(np.linalg.norm(g-exact)/n) if n else None))
        g=z['logit_gradients'][:,0].mean(0);n=float(np.linalg.norm(exact))
        analytic.append(dict(dataset=a['dataset'],seed=a['seed'],t=a['t'],repeat='pooled_four_existing_batches',count_expected_cost=expected,sampled_cost_mean=float(z['costs'][:,:,0].mean()),exact_logit_gradient_norm=n,estimated_norm=float(np.linalg.norm(g)),cosine_to_exact=cosine(g,exact),relative_gradient_error=float(np.linalg.norm(g-exact)/n) if n else None))
        bs=[b for b in batches if b['dataset']==a['dataset'] and int(b['seed'])==a['seed'] and int(b['t'])==a['t']]
        ps=[b for b in pairs if b['dataset']==a['dataset'] and int(b['seed'])==a['seed'] and int(b['t'])==a['t'] and b['group']=='arithmetic_mean' and b['cosine']]
        casedetail.append(dict(dataset=a['dataset'],seed=a['seed'],t=a['t'],seconds_per64=float(np.mean([float(b['score_and_feedback_seconds']) for b in bs])),mean_repeat_cosine=float(np.mean([float(b['cosine']) for b in ps])) if ps else None,unique_graphs_mean=float(np.mean([int(b['unique_hard_graphs']) for b in bs])),exact_one_probabilities=int(bs[0]['exact_one_probability']),exact_zero_probabilities=int(bs[0]['exact_zero_probability'])))
    for ds in ('IMDB','PubMed'):
        bs=[b for b in batches if b['dataset']==ds];secs=np.array([float(b['score_and_feedback_seconds']) for b in bs]);ps=[float(b['cosine']) for b in pairs if b['dataset']==ds and b['group']=='arithmetic_mean' and b['cosine']]
        aa=[b for b in analytic if b['dataset']==ds and b['repeat']!='pooled_four_existing_batches']
        summary.append(dict(dataset=ds,predictions=12,batches64=len(bs),hard_graphs=64*len(bs),seconds_per64_mean=float(secs.mean()),seconds_per64_min=float(secs.min()),seconds_per64_max=float(secs.max()),scoring_seconds_mean=float(np.mean([float(b['scoring_seconds']) for b in bs])),feedback_seconds_mean=float(np.mean([float(b['gradient_seconds']) for b in bs])),extrapolated_3000_scoring_minutes=float(secs.mean()*3000/60),mean_repeat_cosine=float(np.mean(ps)),mean_count_exact_cosine=float(np.mean([b['cosine_to_exact'] for b in aa if b['cosine_to_exact'] is not None])),unique_graphs_per64_mean=float(np.mean([int(b['unique_hard_graphs']) for b in bs]))))
        for name in ('relation_count','joint_degree','present_triangles','absent_triangles','components'):
            gs=[g for g in groups if g['dataset']==ds and g['group']==name];ps=[float(b['cosine']) for b in pairs if b['dataset']==ds and b['group']==name and b['cosine']]
            component.append(dict(dataset=ds,group=name,batches=len(gs),zero_gradient_batches=sum(float(g['gradient_norm'])==0 for g in gs),cost_mean=float(np.mean([float(g['cost_mean']) for g in gs])),gradient_norm_mean=float(np.mean([float(g['gradient_norm']) for g in gs])),norm_share_mean=float(np.mean([float(g['norm_share']) for g in gs])),repeat_cosine_mean=float(np.mean(ps)) if ps else None,nonzero_repeat_pairs=len(ps),relation_common_squared_fraction_mean=float(np.mean([float(g['relation_common_squared_fraction']) for g in gs if float(g['gradient_norm'])>0])) if any(float(g['gradient_norm'])>0 for g in gs) else None))
    old=json.loads((ROOT/'results/hard_clean_pilot_v2_20261110/bindings.json').read_text())
    for p,h in old.items():
        if p.endswith(('/worker.py','/runtime.py','/objective.py')):
            sourceaudit.append(dict(path=p,expected_sha256=h,current_sha256=sha(p),matches=sha(p)==h))
    for n,r in [('summary.csv',summary),('component_summary.csv',component),('analytic_count_reference.csv',analytic),('case_summary.csv',casedetail)]:write(n,r)
    z=dict(summary=summary,components=component,source_binding_checks=sourceaudit,all_draws_reconstructed=True,analytic_reference_scope='relation-count risk only under independent Bernoulli clean sampling; supplementary deterministic analysis, not a change to frozen panel or loss; cannot assess full objective or parameter gradient reliability',seconds=time.time()-start,GPU_hours=0,host=platform.node(),python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,summary_script_sha256=sha(__file__))
    (OUT/'summary.json').write_text(json.dumps(z,indent=2,ensure_ascii=False,allow_nan=False));print(json.dumps(z,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
