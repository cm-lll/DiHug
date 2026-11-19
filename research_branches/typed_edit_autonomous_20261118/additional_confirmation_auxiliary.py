"""Frozen workload and conditionality-aware diversity on all confirmation endpoints."""
import auxiliary_evaluation as old
from collections import defaultdict
import argparse
from pathlib import Path
import json,time,numpy as np
ROOT=old.ROOT;BASE=old.R/'additional_confirmation_v1';OUT=None;app=old.app;legacy=old.legacy
def put(name,value):app.atomic(OUT/name,value)

def main(ds):
    global OUT
    OUT=BASE/f'auxiliary/{ds}'
    source=BASE/f'deploy/{ds}';receipt=app.read(source/'acceptance.json');assert receipt['passed']
    for path,h in app.read(source/'manifest.json')['files'].items():assert app.sha(path)==h,path
    for path,h in receipt['outputs'].items():assert app.sha(path)==h,path
    OUT.mkdir(parents=True,exist_ok=False)
    summary=app.read(source/'summary.json');final=[r for r in summary['rows'] if r['steps']==2048]
    assert len(final)<=18;rows=[]
    for r in final:
        rows.append(dict(dataset=r['dataset'],start_id=r['seed'],producer=r['producer'],arm=r['arm'],repeat=r['repeat'],steps=2048,path=str(source/f"outputs/{r['seed']}_{r['producer']}/{r['arm']}_{r['repeat']}/graph2048.json")))
    for ds in [ds]:
        for sid in [15001,15002]:
            for producer in ['C','D']:
                rows.append(dict(dataset=ds,start_id=sid,producer=producer,arm='zero',repeat=0,steps=0,path=str(BASE/f'prefix/{ds}/outputs/{sid}_{producer}/graph.json')))
    refs={ds:ROOT/f'results/typed_edit_complexity_20261116/data/{ds}/graph.json' for ds in [ds]}
    files=[Path(__file__),Path(__file__).with_name('ADDITIONAL_CONFIRMATION_AUXILIARY.md'),Path(old.__file__),source/'manifest.json',source/'acceptance.json',source/'summary.json',BASE/f'prefix/{ds}/acceptance.json',old.APP/'r8_application_rebound.py',old.APP/'r6_metrics.py',old.APP/'review_endpoints.py',*refs.values(),*[Path(r['path']) for r in rows]]
    bindings={str(p):app.sha(p) for p in files};put('manifest.json',dict(inputs=bindings,rows=rows,new_fits=0,new_graphs=0))
    app.gate(OUT);cost=30.;eng=[]
    for ds,path in refs.items():
        graph=app.read(path);tic=time.monotonic();v,c,a=app.evaluate(graph);checks=app.independent(graph,v,c,a)
        elapsed=time.monotonic()-tic;cost+=2*(len(rows)+3)*elapsed;eng.append(dict(dataset=ds,seconds=elapsed,checks=checks))
    put('engineering.json',dict(passed=cost<=1200,cost_bound_seconds=cost,rows=eng));assert cost<=1200
    tic=time.monotonic();references={};converted={};noise={};outputs=[];groups=defaultdict(list)
    for ds,path in refs.items():
        graph=app.read(path);v,c,a=app.evaluate(graph);app.independent(graph,v,c,a)
        references[ds]=(graph,v,c);converted[ds]=legacy.convert(graph);noise[ds]=[]
        for panel in [1,2]:
            vv,cc,_=app.evaluate(graph,panel);noise[ds].append(dict(panel=panel,distances=app.distances(vv,cc,v,c)))
    for i,row in enumerate(rows):
        assert time.monotonic()-tic<1200
        ds=row['dataset'];graph=app.read(row['path']);reference,rv,rc=references[ds]
        assert graph['node_types']==reference['node_types'] and graph['type_names']==reference['type_names']
        v,c,a=app.evaluate(graph);checks=app.independent(graph,v,c,a);behavior=app.distances(v,c,rv,rc)
        conv=legacy.convert(graph);overlap=len(conv[2]&converted[ds][2])
        rr=dict(**row,sha256=bindings[row['path']],behavior=behavior,workload_calls=len(v),naive_checks=checks,labeled_reference_overlap=overlap/max(1,len(converted[ds][2])),labeled_generated_overlap=overlap/max(1,len(conv[2])))
        outputs.append(rr);groups[ds,row['producer'],row['arm'],row['repeat']].append((rr,conv))
        put(f'outputs/{i}/report.json',rr);np.savez_compressed(OUT/f'outputs/{i}/sampler.npz',values=v);put(f'outputs/{i}/contexts.json',c)
    summaries=[]
    for (ds,producer,arm,repeat),pairs in groups.items():
        assert len(pairs) in [1,2]
        vv=[r['behavior']['mean_behavior_error'] for r,g in pairs];div,cert=legacy.diversity([g for r,g in pairs],converted[ds],limit=2)
        summaries.append(dict(dataset=ds,producer=producer,arm=arm,repeat=repeat,outputs=len(pairs),mean_behavior_error=float(np.mean(vv)),across_start_sample_sd=float(np.std(vv,ddof=1)) if len(vv)>1 else None,diversity=div,certificates=cert))
    aggregate=[]
    for ds in refs:
        for producer,arms in [('C',['F','Q','N','NS','zero']),('D',['N','zero'])]:
            for arm in arms:
                ss=[r for r in summaries if (r['dataset'],r['producer'],r['arm'])==(ds,producer,arm)]
                if arm in ['F','Q']:values=[r['mean_behavior_error'] for r in ss];assert len(values) in [0,3];unit='three bank means over eligible starts'
                else:values=[r['behavior']['mean_behavior_error'] for r in outputs if (r['dataset'],r['producer'],r['arm'])==(ds,producer,arm)];assert len(values)<=2;unit='starts'
                aggregate.append(dict(dataset=ds,producer=producer,arm=arm,mean=float(np.mean(values)) if values else None,sample_sd=float(np.std(values,ddof=1)) if len(values)>1 else None,sd_unit=unit))
    put('per_output.json',outputs);put('summary.json',dict(groups=summaries,aggregate=aggregate,reference_panel_noise=noise,not_fixed_target_diversity=True,not_general_utility=True,model_selection=False))
    for path,h in bindings.items():assert app.sha(path)==h,path
    put('acceptance.json',dict(passed=True,evaluated_states=len(rows),groups=len(summaries),new_fits=0,new_graphs=0,workload_calls=sum(r['workload_calls'] for r in outputs),independent_naive_checks=sum(r['naive_checks'] for r in outputs),seconds=time.monotonic()-tic))
    print('Confirmation auxiliary evaluation accepted:',len(rows),'states',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('dataset',choices=['acm','freebase']);args=parser.parse_args()
    try:main(args.dataset)
    except BaseException as exc:
        if OUT is not None and OUT.exists():put('failure.json',dict(type=type(exc).__name__,message=str(exc)))
        raise
