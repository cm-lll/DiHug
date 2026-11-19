"""Unchanged workload/nonisomorphism probe on every accepted new endpoint."""
import argparse,json,time,statistics
from collections import defaultdict
from pathlib import Path
import auxiliary_evaluation as old
ROOT=old.ROOT;BASE=ROOT/'results/typed_edit_crossdomain_augmentation_20261118';app=old.app;legacy=old.legacy

def main(ds):
    source=BASE/f'confirmation/deploy/{ds}';accept=app.read(source/'acceptance.json');assert accept['passed']
    for p,h in app.read(source/'manifest.json')['files'].items():assert app.sha(p)==h,p
    for p,h in accept['outputs'].items():assert app.sha(p)==h,p
    out=BASE/f'confirmation/auxiliary/{ds}';out.mkdir(parents=True,exist_ok=False)
    def put(name,v):app.atomic(out/name,v)
    final=[r for r in app.read(source/'summary.json')['rows'] if r['steps']==2048];rows=[]
    for r in final:rows.append(dict(dataset=ds,start_id=r['seed'],producer=r['producer'],arm=r['arm'],bank=r['bank'],steps=2048,path=str(source/f"outputs/{r['seed']}_{r['producer']}/{r['arm']}_{r['bank']}/graph2048.json")))
    prefix=BASE/f'confirmation/prefix/{ds}'
    for r in app.read(prefix/'summary.json')['rows']:
        rows.append(dict(dataset=ds,start_id=r['seed'],producer=r['arm'],arm='zero',bank=0,steps=0,path=str(prefix/f"outputs/{r['seed']}_{r['arm']}/graph.json")))
    ref=ROOT/f'results/typed_edit_complexity_20261116/data/{ds}/graph.json'
    files=[Path(__file__),Path(__file__).with_name('CROSSDOMAIN_AUGMENTATION_CONFIRMATION.md'),Path(old.__file__),source/'manifest.json',source/'acceptance.json',source/'summary.json',prefix/'acceptance.json',old.APP/'r8_application_rebound.py',old.APP/'r6_metrics.py',old.APP/'review_endpoints.py',ref,*[Path(r['path']) for r in rows]]
    bindings={str(p):app.sha(p) for p in files};put('manifest.json',dict(inputs=bindings,rows=rows,new_fits=0,new_graphs=0,model_selection=False));app.gate(out)
    g=app.read(ref);tic=time.monotonic();rv,rc,ra=app.evaluate(g);app.independent(g,rv,rc,ra);cost=30+2*(len(rows)+1)*(time.monotonic()-tic)
    put('engineering.json',dict(passed=cost<=1200,cost_bound_seconds=cost));assert cost<=1200;tic=time.monotonic();refconv=legacy.convert(g);outputs=[];groups=defaultdict(list)
    for i,row in enumerate(rows):
        assert time.monotonic()-tic<1200
        graph=app.read(row['path']);assert graph['node_types']==g['node_types'] and graph['type_names']==g['type_names']
        v,c,a=app.evaluate(graph);checks=app.independent(graph,v,c,a);behavior=app.distances(v,c,rv,rc);conv=legacy.convert(graph)
        r=dict(**row,sha256=bindings[row['path']],behavior=behavior,workload_calls=len(v),naive_checks=checks,labeled_reference_overlap=len(conv[2]&refconv[2])/max(1,len(refconv[2])))
        outputs.append(r);groups[row['producer'],row['arm'],row['bank']].append((r,conv));put(f'outputs/{i}/report.json',r)
    pairs=[]
    for (producer,arm,bank),items in groups.items():
        values=[r['behavior']['mean_behavior_error'] for r,g in items];div,cert=legacy.diversity([g for r,g in items],refconv,limit=2)
        pairs.append(dict(producer=producer,arm=arm,bank=bank,outputs=len(items),mean_behavior_error=statistics.mean(values),across_start_sd=statistics.stdev(values) if len(values)>1 else None,diversity=div,certificates=cert))
    aggregate=[]
    for producer,arms in [('C',['T','U','B','Q','N','NS','zero']),('D',['N','zero'])]:
        for arm in arms:
            if arm in ['T','U','B','Q']:values=[r['mean_behavior_error'] for r in pairs if r['producer']==producer and r['arm']==arm];unit='paired-bank means over available starts'
            else:values=[r['behavior']['mean_behavior_error'] for r in outputs if r['producer']==producer and r['arm']==arm];unit='starts'
            aggregate.append(dict(producer=producer,arm=arm,mean=statistics.mean(values) if values else None,sample_sd=statistics.stdev(values) if len(values)>1 else None,sd_unit=unit))
    for p,h in bindings.items():assert app.sha(p)==h,p
    put('per_output.json',outputs);put('summary.json',dict(aggregate=aggregate,groups=pairs,not_fixed_target_diversity=True,not_general_utility=True,model_selection=False));put('acceptance.json',dict(passed=True,evaluated_states=len(outputs),groups=len(pairs),workload_calls=sum(r['workload_calls'] for r in outputs),independent_naive_checks=sum(r['naive_checks'] for r in outputs),seconds=time.monotonic()-tic,new_graphs=0,new_fits=0));print(ds,'auxiliary accepted',aggregate,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset',choices=['pubmed','acm']);main(p.parse_args().dataset)
