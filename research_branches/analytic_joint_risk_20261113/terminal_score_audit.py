"""CPU-only same-objective audit of existing 80 final graphs, no model import.

Exact original frozen finite-frequency target; no frequency/scale adjustment.
The four-sample bias/variance decomposition is descriptive, not an independent
estimate of training repeats or of the conditional Bernoulli variance formula.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import json,csv,time,hashlib,itertools,statistics,resource,signal
from pathlib import Path
import numpy as np
from core import hard_losses,risks
import torch

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'results/analytic_joint_generation_run_20261113T225800'
SHORT=ROOT/'results/analytic_joint_short_20261113'
TARGET=ROOT/'results/analytic_joint_risk_20261113'
OUT=ROOT/'results/analytic_joint_terminal_score_audit_20261114'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def table(path,rows):
    path.with_suffix('.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False,allow_nan=False))
    if rows:
        with path.with_suffix('.csv').open('w') as f:
            w=csv.DictWriter(f,list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def main():
    start=time.time();cpu=time.process_time();OUT.mkdir(exist_ok=False)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('600s CPU audit wall cap')));signal.alarm(600)
    def check():
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>4*1024*1024:raise MemoryError('4GiB RSS cap')
    bindings={}
    def bind(path,expected=None):
        path=Path(path).resolve();h=sha(path)
        if expected is not None:assert h==expected,str(path)
        bindings[str(path)]=h;return path
    def read(path):return json.loads(bind(path).read_text())
    def write(name,value):(OUT/name).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False))
    bind(__file__);bind(Path(__file__).with_name('core.py'))
    p=read(SOURCE/'protocol.json');inventory=read(SOURCE/'inventory.json');assert len(inventory)==80
    oldbindings=read(SOURCE/'asset_recheck.json')['bindings']
    assert sha(Path(__file__).with_name('core.py'))==oldbindings[str(Path(__file__).with_name('core.py'))]
    sp=read(SHORT/'protocol.json');generation_rows=read(SOURCE/'per_graph.json')
    write('protocol.json',dict(scope='existing80 final graphs and cached fixed-input scores; original joint-degree CF target only',
        start_epoch=start,wall_cap_seconds=600,CPU_threads=1,RSS_cap_GiB=4,model_calls=0,GPU_hours=0,new_graphs=0,
        graph_count=80,targets=['IMDB','PubMed'],arms=['parent','bias','risk'],candidate_readouts=['raw','ema'],
        rule='all registered graphs; no selection by score; fixed original32 frequencies/scales/equal type weighting',
        interpretation='conditional forward-noise risk and full-chain empirical hard loss are same per-graph functional under different input/output distributions; four samples descriptive only',
        decomposition='empirical mean hard loss = error of mean CF + mean squared CF deviation, denominator4 not unbiased variance denominator3'))
    contexts={};checks=[];cf={};graphrows=[];typerows=[];frequencies=[];state_inventory=[]
    for row in p['parents']:
        ds=row['dataset']
        if ds in contexts:continue
        z=np.load(bind(row['data'],oldbindings[row['data']]));types=z['types'];schema=z['schema'];edges=z['edges'];edges=edges if edges.shape[0]==2 else edges.T
        frozen_path=TARGET/f'{ds}_frozen_target.json';bind(frozen_path,sp['bindings'][str(frozen_path)])
        frozen=read(frozen_path);assert np.array_equal(schema,frozen['payload']['profile']['family_pairs'])
        q=np.array(list(itertools.combinations(range(len(types)),2))).T
        lookup={tuple(pair):i for i,pair in enumerate(schema)};family=np.array([lookup[tuple(sorted(types[e]))] for e in q.T])
        slot={tuple(e):i for i,e in enumerate(q.T)};ref=np.zeros(len(family))
        for e in edges.T:ref[slot[tuple(sorted(e))]]=1
        counts=np.bincount(family,weights=ref,minlength=len(schema));capacities=np.bincount(family,minlength=len(schema))
        contexts[ds]=dict(types=types,schema=schema,frozen=frozen,q=q,family=family,slot=slot,reference=ref,
            type_names=row['config']['datasets'][ds]['type_names'],fixed_zero=counts==0,fixed_one=counts==capacities)
        assert abs(hard_losses(ref[None,:],types,q,family,frozen)[0])<1e-25
        checks.append(dict(dataset=ds,reference_hard_loss=float(hard_losses(ref[None,:],types,q,family,frozen)[0]),
            schema_matches=True,frozen_target_unchanged=True,frequency_count=32))
    for item in inventory:
        check();sc=contexts[item['dataset']];types=sc['types'];q=sc['q'];family=sc['family'];frozen=sc['frozen'];directory=Path(item['directory'])
        summary=read(directory/'summary.json');gp=bind(item['graph'],summary['graph_sha256']);z=np.load(gp)
        assert np.array_equal(z['types'],types) and np.array_equal(z['schema'],sc['schema'])
        e=z['edges'];assert e.shape[0]==2 and np.all(e[0]<e[1]) and len(set(map(tuple,e.T)))==e.shape[1]
        bits=np.zeros(len(family));
        for edge in e.T:bits[sc['slot'][tuple(edge)]]=1
        assert not bits[sc['fixed_zero'][family]].any() and np.all(bits[sc['fixed_one'][family]]==1)
        degree=np.zeros((len(types),len(sc['schema'])))
        for endpoint in q:np.add.at(degree,(endpoint,family),bits)
        phis=[];parts=[];ident={k:item[k] for k in ('model','dataset','train_seed','arm','readout','sample_seed')}
        for typ in sorted(np.unique(types)):
            key=str(typ);w=np.array(frozen['payload']['omega'][key]);t=frozen['payload']['target_cf'][key];target=np.array(t['real'])+1j*np.array(t['imag'])
            value=np.exp(1j*(degree[types==typ]@w.T)).mean(0);err=abs(value-target)**2
            phis.append(value);parts.append(float(err.mean()))
            typerows.append(dict(**ident,type=int(typ),type_name=sc['type_names'][typ],nodes=int((types==typ).sum()),hard_score=float(err.mean())))
            for k,loss in enumerate(err):frequencies.append(dict(**ident,type=int(typ),frequency_index=k,squared_error=float(loss)))
        score=float(np.mean(parts));independent=float(hard_losses(bits[None,:],types,q,family,frozen)[0])
        assert abs(score-independent)<1e-14
        # Four deterministic hard endpoints checked against original analytical function.
        if item['arm']=='parent' and item['sample_seed']==next(r for r in inventory if r['model']==item['model'])['sample_seed']:
            vals,_=risks(torch.tensor(bits,dtype=torch.float64),types,q,family,frozen)
            assert abs(float(vals['variance']))<1e-12 and abs(float(vals['risk'])-score)<1e-12 and abs(float(vals['bias'])-score)<1e-12
            checks.append(dict(model=item['model'],sample_seed=item['sample_seed'],hard_score=score,
                analytic_hard_risk=float(vals['risk']),analytic_hard_variance=float(vals['variance']),passed=True))
        metrics=next(x for x in generation_rows if x['view']=='whole' and all(x[k]==item[k] for k in ('model','arm','readout','sample_seed')))
        graphrows.append(dict(**ident,hard_score=score,graph_sha256=sha(gp),**{k:metrics[k] for k in
            ('edges','degree_W1','joint_TV','joint_relation_degree_L1','triangle_abs_error','triangle_composition_TV','MP2_TV','MP3_TV','MP3_abs_error','LCC','low_fraction_error','hub_top10_mean_error')}))
        cf[(item['model'],item['arm'],item['readout'],item['sample_seed'])]=np.array(phis)
        # Presence-only audit: do not infer unrecorded intermediate graph states.
        first=json.loads((directory/'steps.jsonl').read_text().splitlines()[0])
        state_inventory.append(dict(**ident,intermediate_Xt_files=len(list(directory.glob('Xt_*.npz'))),
            step_log_keys=sorted(first),latest_complete_exists=(directory/'latest_complete.pt').exists(),
            intermediate_probabilities_saved=bool(list(directory.glob('*probabilities*.npz')))))
    summaries=[];type_summaries=[];validation=[]
    for row in p['parents']:
        r=read(SHORT/row['id']/'result.json')
        for arm,readout in [('parent','parent_EMA3000'),('bias','raw'),('bias','ema'),('risk','raw'),('risk','ema')]:
            samples=r['baseline'] if arm=='parent' else r['arms'][arm]['evaluations']['16'][readout]
            ident=dict(model=row['id'],dataset=row['dataset'],train_seed=row['train_seed'],arm=arm,readout=readout)
            rr=[a for a in graphrows if all(a[k]==ident[k] for k in ('model','arm','readout'))];assert len(rr)==4
            phis=np.stack([cf[(x['model'],arm,readout,x['sample_seed'])] for x in rr]);sc=contexts[row['dataset']]
            target=np.array([np.array(sc['frozen']['payload']['target_cf'][str(t)]['real'])+1j*np.array(sc['frozen']['payload']['target_cf'][str(t)]['imag']) for t in sorted(np.unique(sc['types']))])
            empirical_bias=float(np.mean(abs(phis.mean(0)-target)**2));empirical_var=float(np.mean(abs(phis-phis.mean(0))**2))
            mean=statistics.mean(x['hard_score'] for x in rr);assert abs(mean-empirical_bias-empirical_var)<1e-14
            summaries.append(dict(**ident,samples=4,hard_score_mean=mean,hard_score_sample_SD=statistics.stdev(x['hard_score'] for x in rr),
                empirical_mean_CF_error=empirical_bias,empirical_CF_variance_denominator4=empirical_var,
                cached_validation_risk_mean=statistics.mean(x['loss']['risk'] for x in samples),
                cached_validation_bias_mean=statistics.mean(x['loss']['bias'] for x in samples),
                cached_validation_variance_mean=statistics.mean(x['loss']['variance'] for x in samples)))
            for t,name in enumerate(sc['type_names']):
                ts=[x for x in typerows if all(x[k]==ident[k] for k in ('model','arm','readout')) and x['type']==t]
                type_summaries.append(dict(**ident,type=t,type_name=name,hard_score_mean=statistics.mean(x['hard_score'] for x in ts)))
            for a in samples:validation.append(dict(**ident,t=a['t'],rep=a['rep'],input_hash=a['input_hash'],**a['loss']))
    comparisons=[];sample_pairs=[];type_pairs=[]
    for r in summaries:
        if r['arm']=='parent':continue
        for against in ('parent','bias'):
            if against=='bias' and r['arm']!='risk':continue
            base=next(x for x in summaries if x['model']==r['model'] and x['arm']==against and (against=='parent' or x['readout']==r['readout']))
            d={k:r[k] for k in ('model','dataset','train_seed','arm','readout')};d['against']=against
            for k in ('hard_score_mean','empirical_mean_CF_error','empirical_CF_variance_denominator4','cached_validation_risk_mean','cached_validation_bias_mean'):
                d[k+'_delta']=r[k]-base[k];d[k+'_relative_percent']=100*(r[k]/base[k]-1) if base[k]!=0 else None
            comparisons.append(d)
            for a in [x for x in graphrows if x['model']==r['model'] and x['arm']==r['arm'] and x['readout']==r['readout']]:
                b=next(x for x in graphrows if x['model']==a['model'] and x['arm']==against and x['readout']==base['readout'] and x['sample_seed']==a['sample_seed'])
                sample_pairs.append(dict(**{k:d[k] for k in ('model','dataset','train_seed','arm','readout','against')},sample_seed=a['sample_seed'],
                    **{k+'_delta':a[k]-b[k] for k in ('hard_score','degree_W1','joint_TV','joint_relation_degree_L1','triangle_abs_error','MP3_TV','MP3_abs_error','low_fraction_error','hub_top10_mean_error','LCC')}))
            for t in contexts[r['dataset']]['type_names']:
                a=next(x for x in type_summaries if x['model']==r['model'] and x['arm']==r['arm'] and x['readout']==r['readout'] and x['type_name']==t)
                b=next(x for x in type_summaries if x['model']==r['model'] and x['arm']==against and x['readout']==base['readout'] and x['type_name']==t)
                type_pairs.append(dict(**{k:d[k] for k in ('model','dataset','train_seed','arm','readout','against')},type_name=t,delta=a['hard_score_mean']-b['hard_score_mean']))
    for name,rr in [('per_graph',graphrows),('per_type',typerows),('per_frequency',frequencies),('model_summary',summaries),('type_summary',type_summaries),
        ('cached_validation',validation),('comparisons',comparisons),('paired_samples',sample_pairs),('paired_types',type_pairs),('state_inventory',state_inventory)]:table(OUT/name,rr)
    write('checks.json',checks)
    assert all(sha(path)==h for path,h in bindings.items())
    write('asset_bindings.json',bindings)
    write('completion.json',dict(status='complete',graphs=80,model_readouts=20,source_files=len(bindings),
        model_calls=0,new_graphs=0,GPU_hours=0,CPU_seconds=time.process_time()-cpu,wall_seconds=time.time()-start,
        RSS_peak_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,original_assets_unchanged=True,epoch=time.time()))
    for r in comparisons:
        if r['arm']=='risk' and r['against']=='parent':print(json.dumps(r))
    signal.alarm(0)
if __name__=='__main__':main()
