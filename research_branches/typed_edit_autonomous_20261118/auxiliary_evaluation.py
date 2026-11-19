"""Frozen R8 workload and R6 diversity diagnostics on accepted typed-edit graphs."""
import sys,json,csv,hashlib,time
from pathlib import Path
from collections import defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'results/typed_edit_autonomous_20261118';OUT=R/'r28_auxiliary'
APP=ROOT.parent/'DiHug_paper_revision/results/r8_application_20261015_v2/source/scripts';sys.path.insert(0,str(APP))
import r8_application_rebound as app
import r6_metrics as legacy

def put(name,v):app.atomic(OUT/name,v)
def inventory():
    rows=[]
    for ds,starts in [('pubmed',[106,107]),('imdb',[108,109])]:
        for sid in starts:
            zero=ROOT/f'results/typed_edit_complexity_20261116/prefixes/pubmed/seed{sid}/initial.json' if ds=='pubmed' else R/f'r22_prefix/outputs/{sid}_C/graph.json'
            rows.append(dict(dataset=ds,start_id=sid,arm='zero',steps=0,path=str(zero)))
            for arm in ['F','Q','N','NS']:
                if arm=='F':folder=R/f'r27_deploy/outputs/{ds}/{sid}'
                elif ds=='imdb':folder=R/f'r24_complete/outputs/{sid}_C/{arm}'
                elif arm=='Q':folder=R/f'r18_deploy/outputs/pubmed/start{sid}/Q'
                else:folder=R/f'r16_deploy/outputs/start{sid}/{arm}'
                for step in [512,2048]:
                    path=folder/'graph512.json' if step==512 else R/f'r28_extend/outputs/{ds}/{sid}/{arm}/graph2048.json'
                    rows.append(dict(dataset=ds,start_id=sid,arm=arm,steps=step,path=str(path)))
    return rows

def main():
    assert app.read(R/'r28_extend/acceptance.json')['passed'] and app.read(R/'r27_finalize/acceptance.json')['passed'];OUT.mkdir(parents=True,exist_ok=False)
    rows=inventory();assert len(rows)==36;refs={ds:ROOT/f'results/typed_edit_complexity_20261116/data/{ds}/graph.json' for ds in ['pubmed','imdb']}
    paths=[Path(__file__),Path(__file__).with_name('AUXILIARY_EVALUATION.md'),R/'r28_extend/acceptance.json',R/'r27_finalize/acceptance.json',APP/'r8_application_rebound.py',APP/'r6_metrics.py',APP/'review_endpoints.py',*refs.values(),*[Path(r['path']) for r in rows]]
    bindings={str(p):app.sha(p) for p in paths};put('manifest.json',dict(inputs=bindings,rows=rows,new_fits=0,new_graphs=0));app.gate(OUT);cost=30.;eng=[]
    for ds in refs:
        g=app.read(refs[ds]);tic=time.monotonic();v,c,a=app.evaluate(g);n=app.independent(g,v,c,a);elapsed=time.monotonic()-tic;cost+=2*21*elapsed;eng.append(dict(dataset=ds,seconds=elapsed,calls=len(v),independent_checks=n))
    put('engineering.json',dict(passed=cost<=1200,cost_bound_seconds=cost,rows=eng));assert cost<=1200;tic=time.monotonic();references={};noise={};converted={};outputs=[];groups=defaultdict(list)
    for ds,path in refs.items():
        g=app.read(path);v,c,a=app.evaluate(g);app.independent(g,v,c,a);references[ds]=(g,v,c);converted[ds]=legacy.convert(g);noise[ds]=[]
        for panel in [1,2]:vv,cc,_=app.evaluate(g,panel);noise[ds].append(dict(panel=panel,distances=app.distances(vv,cc,v,c)))
    for i,row in enumerate(rows):
        assert time.monotonic()-tic<1200
        ds=row['dataset'];g=app.read(row['path']);ref,rv,rc=references[ds];assert g['node_types']==ref['node_types'] and g['type_names']==ref['type_names']
        v,c,a=app.evaluate(g);checks=app.independent(g,v,c,a);behavior=app.distances(v,c,rv,rc);conv=legacy.convert(g);refconv=converted[ds];overlap=len(conv[2]&refconv[2])
        rr=dict(**row,sha256=bindings[row['path']],behavior=behavior,workload_calls=len(v),naive_checks=checks,labeled_reference_overlap=overlap/max(1,len(refconv[2])),labeled_generated_overlap=overlap/max(1,len(conv[2])))
        outputs.append(rr);groups[(ds,row['steps'],row['arm'])].append((rr,conv));put(f'outputs/{i}/report.json',rr);np.savez_compressed(OUT/f'outputs/{i}/sampler.npz',values=v);put(f'outputs/{i}/contexts.json',c);print('aux evaluated',ds,row['start_id'],row['arm'],row['steps'],round(behavior['mean_behavior_error'],6),flush=True)
    summary=[]
    for (ds,step,arm),pairs in groups.items():
        vv=[r['behavior']['mean_behavior_error'] for r,g in pairs];div,cert=legacy.diversity([g for r,g in pairs],converted[ds],limit=2)
        summary.append(dict(dataset=ds,steps=step,arm=arm,outputs=len(vv),mean_behavior_error=float(np.mean(vv)),across_start_sample_sd=float(np.std(vv,ddof=1)),training_seed_sd=None,diversity=div,certificates=cert))
    put('per_output.json',outputs);put('summary.json',dict(groups=summary,reference_panel_noise=noise,caveats=['Fixed two-hop workload proxy, not supervised accuracy or general utility.','Two outputs per group do not establish population diversity.','Labeled overlap depends on node identities; no privacy guarantee.','Unresolved typed isomorphism cases retain interval bounds.','Not used to select G versus F.']))
    for p,h in bindings.items():assert app.sha(p)==h,p
    put('acceptance.json',dict(passed=True,evaluated_states=36,groups=len(summary),new_fits=0,new_graphs=0,workload_calls=sum(r['workload_calls'] for r in outputs),independent_naive_checks=sum(r['naive_checks'] for r in outputs),seconds=time.monotonic()-tic))

if __name__=='__main__':
    try:main()
    except BaseException as exc:put('failure.json',dict(type=type(exc).__name__,message=str(exc)));raise
