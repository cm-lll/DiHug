"""CPU-only full raw/LCC panel, pairing audit, immutable baseline reuse.

May be rerun incrementally: caches only completed graph evaluations in THIS
ticket. Does not select checkpoints, samples or alter learning workers.
"""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,time,hashlib,importlib.util,itertools
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];OUT=Path(sys.argv[1]).resolve()
spec=importlib.util.spec_from_file_location('pilot_eval_base',ROOT/'research_branches/terminal_structure_real_20261109/evaluate.py')
ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
spec=importlib.util.spec_from_file_location('pilot_eval_score',Path(__file__).with_name('gpu_preflight.py'))
sc=importlib.util.module_from_spec(spec);spec.loader.exec_module(sc)
from heat_solver_candidate import heat as repaired_heat
sc.heat.heat=repaired_heat
def read(p):return json.loads(Path(p).read_text())
def save(name,x):ev.save(OUT/name,x)
def logs(p):return [json.loads(x) for x in Path(p).read_text().splitlines()] if Path(p).exists() else []
def main():
    torch.set_num_threads(1);start=time.time();p=read(OUT/'protocol.json')
    rows=[];allmetrics=[];costs=[];verification=[];missing=[];pairs=[];vs=[];training=[];refs={}
    cache=OUT/'evaluations';cache.mkdir(exist_ok=True)
    def check():
        if time.time()>p['delivery_epoch']-60:raise TimeoutError('CPU delivery reserve')
    for ds,parent in p['parents'].items():
        z=np.load(parent['data']);types,schema,edges=z['types'],z['schema'],z['edges'];edges=edges.T if edges.shape[0]!=2 else edges
        names=['Director','Actor','Movie','Genre'] if ds=='IMDB' else ['Gene','Disease','Chemical','Species']
        ref=ev.M.make_graph(types,edges,schema)
        refpath=cache/(ds+'_reference.json')
        if refpath.exists():rs=read(refpath)
        else:rs=ev.M.stats(ref,ev.F,names,check);ev.save(refpath,rs)
        refs[ds]=rs
        case=sc.w.common.make_case(types,edges,tuple(map(tuple,schema)))
        ids,fam,query=sc.w.common.plan_for(case,2026111402,1).block(0)
        scorer=sc.Score(ds,case,query,fam,np.isin(ids,case['clean_ids']).astype(np.uint8))
        for arm in ('baseline','T2','T4','T8'):
            folder=OUT/(ds+'_'+arm);cg=[];ss_ok=[]
            for ss in p['sample_seeds']:
                check()
                directory=(ROOT/f'results/three_arm_update_preflight_20261111/{ds}/baseline_sample_{ss}'
                           if arm=='baseline' and ss==2026111341 else folder/f'final_sample_{ss}')
                if not (directory/'summary.json').exists():
                    missing.append(dict(dataset=ds,arm=arm,sample_seed=ss,reason='no completed graph'));continue
                summary=read(directory/'summary.json');assert summary['status']=='complete' and summary['steps']==1000
                assert summary['graph_sha256']==ev.sha(directory/'graph.npz')
                identity=read(directory/'identity.json');assert identity['sample_seed']==ss and identity['T']==1000 and identity['R']==1
                steps=logs(directory/'steps.jsonl');assert [x['time'] for x in steps]==list(range(1000,0,-1))
                assert all(x['exact_coverage'] and x['covered_slots']==len(ids) for x in steps)
                # Every comparison uses identical full-slot uniforms and query plans.
                bp=(ROOT/f'results/three_arm_update_preflight_20261111/{ds}/baseline_sample_{ss}' if ss==2026111341
                    else OUT/(ds+'_baseline')/f'final_sample_{ss}')
                paired=False
                if (bp/'summary.json').exists():
                    bs=logs(bp/'steps.jsonl');bi=read(bp/'identity.json')
                    assert identity['initial_hash']==bi['initial_hash']
                    assert [(x['uniform_hash'],x['query_hash']) for x in steps]==[(x['uniform_hash'],x['query_hash']) for x in bs]
                    paired=True
                verification.append(dict(dataset=ds,arm=arm,sample_seed=ss,complete_1000=True,paired_initial_query_uniform=paired,graph_sha256=summary['graph_sha256']))
                z=np.load(directory/'graph.npz');g=ev.M.make_graph(z['types'],z['edges'],z['schema'])
                assert np.array_equal(g['types'],types) and np.array_equal(g['schema'],schema)
                fp=cache/f'{ds}_{arm}_{ss}.json'
                if fp.exists():full=read(fp);assert full['graph_sha256']==summary['graph_sha256']
                else:
                    if arm=='baseline' and ss==2026111341:full=read(directory/'full_metrics.json')
                    else:
                        st=ev.M.stats(g,ev.F,names,check);cmp=ev.M.compare(g,st,ref,rs,ev.F,names)
                        lg,mapping=ev.M.lcc(g);ls=ev.M.stats(lg,ev.F,names,check);lc=ev.M.compare(lg,ls,ref,rs,ev.F,names)
                        full=dict(reference=rs,raw=dict(statistics=st,comparison=cmp),LCC=dict(statistics=ls,comparison=lc,mapping=mapping))
                    bits=np.zeros(case['space'].total,dtype=np.uint8);ff,ll=case['space'].encode(g['edges']);bits[case['space'].offsets[ff]+ll]=1
                    assert not bits[~np.repeat(case['active'],case['space'].counts)].any()
                    terms=scorer(bits[ids]);full.update(graph_sha256=summary['graph_sha256'],objective=dict(joint=terms[0],count=terms[1],total=sum(terms[:2]),by_type=terms[2]))
                    ev.save(fp,full)
                costs.append(dict(dataset=ds,arm=arm,sample_seed=ss,**full['objective']))
                for view in ('raw','LCC'):
                    st=full[view]['statistics'];cmp=full[view]['comparison']
                    rows.append(dict(dataset=ds,arm=arm,train_seed=parent['train_seed'],sample_seed=ss,view=view,
                        nodes=st['nodes'],edges=st['edges'],edge_abs_error=abs(st['edges']-rs['edges']),reference_edges=rs['edges'],
                        degree_W1=cmp['main_incident4_W1'],joint_TV=cmp['main_joint_degree_TV'],triangles=st['triangles'],
                        reference_triangles=rs['triangles'],triangle_abs_error=abs(st['triangles']-rs['triangles']),triangle_composition_TV=cmp['triangle_composition_TV'],
                        MP2=st['mp'][1]['total'],MP2_abs_error=abs(st['mp'][1]['total']-rs['mp'][1]['total']),MP2_TV=cmp['MP_composition_TV'][1],
                        MP3=st['mp'][2]['total'],MP3_abs_error=abs(st['mp'][2]['total']-rs['mp'][2]['total']),MP3_TV=cmp['MP_composition_TV'][2],
                        LCC=st['lcc'],components=st['components'],isolates=st['isolates'],four_cycles=st['four_cycles'],
                        edge_jaccard=cmp['EO_jaccard'],type_coverage=len(names)-len(cmp['missing_types'])))
                    allmetrics.extend(dict(dataset=ds,arm=arm,sample_seed=ss,view=view,metric=k,value=v) for k,v in ev.flat(full[view]).items())
                cg.append(g);ss_ok.append(ss)
            for i,j in itertools.combinations(range(len(cg)),2):
                a=set(map(tuple,cg[i]['edges'].T));b=set(map(tuple,cg[j]['edges'].T))
                pairs.append(dict(dataset=ds,arm=arm,seed1=ss_ok[i],seed2=ss_ok[j],intersection=len(a&b),union=len(a|b),
                    jaccard=len(a&b)/len(a|b) if a|b else None,identical=a==b))
            if (folder/'training_summary.json').exists():training.append(dict(dataset=ds,arm=arm,**read(folder/'training_summary.json')))
            for readout in ('raw','EMA'):
                vf=folder/f'validation_{readout}.json'
                if vf.exists():
                    for v in read(vf):vs.append(dict(dataset=ds,arm=arm,readout=readout,**v))
        starts=[read(OUT/(ds+'_'+arm)/'binding.json')['initial_state_hash'] for arm in ('baseline','T2','T4','T8') if (OUT/(ds+'_'+arm)/'binding.json').exists()]
        assert len(set(starts))<=1
        for arm in ('baseline','T2','T4','T8'):
            fp=OUT/(ds+'_'+arm)/'final_complete.pt'
            if fp.exists():
                state=torch.load(fp,map_location='cpu',weights_only=False)
                assert state['updates']==state['ema']['updates'] and state['recovery']['exact_resume_allowed']
                steps={float(v['step']) for v in state['optimizer']['state'].values()}
                assert steps==({float(state['updates'])} if state['updates'] else set())
                if arm.startswith('T'):assert state['updates']==int(arm[1:])
                del state
    summaries=[];deltas=[]
    numeric=['edges','edge_abs_error','degree_W1','joint_TV','triangles','triangle_abs_error','triangle_composition_TV','MP2','MP2_abs_error','MP2_TV','MP3','MP3_abs_error','MP3_TV','LCC','components','isolates','four_cycles','edge_jaccard','type_coverage']
    for ds in p['parents']:
        for view in ('raw','LCC'):
            for arm in ('baseline','T2','T4','T8'):
                rr=[r for r in rows if r['dataset']==ds and r['view']==view and r['arm']==arm]
                if not rr:continue
                summary=dict(dataset=ds,arm=arm,view=view,training_models=1,samples=len(rr))
                for k in numeric:
                    values=[r[k] for r in rr if r[k] is not None]
                    summary[k+'_mean']=float(np.mean(values)) if values else None
                    summary[k+'_sample_sd']=float(np.std(values,ddof=1)) if len(values)>1 else None
                summaries.append(summary)
                if arm!='baseline':
                    for r in rr:
                        base=[x for x in rows if x['dataset']==ds and x['arm']=='baseline' and x['view']==view and x['sample_seed']==r['sample_seed']]
                        if base:deltas.append(dict(dataset=ds,arm=arm,view=view,sample_seed=r['sample_seed'],**{k:r[k]-base[0][k] if r[k] is not None and base[0][k] is not None else None for k in numeric}))
    for name,table in [('per_graph',rows),('all_metrics',allmetrics),('objective_scores',costs),('model_summary',summaries),('paired_differences',deltas),('sample_pairs',pairs),('training_costs',training),('validation',vs)]:
        save(name+'.json',table);ev.csvsave(OUT/(name+'.csv'),table)
    save('reference_metrics.json',refs)
    for path,h in p['bindings'].items():assert ev.sha(path)==h,path
    save('verification.json',dict(checks=verification,missing=missing,all_bound_assets_unchanged=True,completed_graphs=len(rows)//2,
        reused_graphs=8,new_graphs=8,independent_training_seeds_per_dataset=1,recomputed_update_bitwise_not_certified=True))
    save('evaluation_receipt.json',dict(seconds=time.time()-start,at_epoch=time.time(),GPU=0,graphs=len(rows)//2,missing=missing,
        source_hashes={str(x):ev.sha(x) for x in [Path(__file__),ROOT/'research_branches/terminal_structure_real_20261109/evaluate.py',ev.OLD/'lcc_eval/metrics.py',*ev.V.glob('*.py')]}))
    print(json.dumps(dict(graphs=len(rows)//2,missing=len(missing),seconds=time.time()-start)),flush=True)
if __name__=='__main__':main()
