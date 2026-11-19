"""CPU-only full original/LCC metrics, paired RNG audit and sample dispersion."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,time,importlib.util,itertools,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'research_branches/one_step_vs_terminal_plan_20261111'))
spec=importlib.util.spec_from_file_location('fixed_evaluator',ROOT/'research_branches/terminal_structure_real_20261109/evaluate.py')
ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
import gpu_preflight as sc
from heat_solver_candidate import heat
sc.heat.heat=heat
def read(p):return json.loads(Path(p).read_text())
def main():
    out=Path(sys.argv[1]);p=read(out/'protocol.json');start=time.time();ev.torch.set_num_threads(1)
    cache=out/'evaluation';cache.mkdir(exist_ok=True)
    rows=[];all_metrics=[];costs=[];pairs=[];missing=[];verification=[];trajectories=[];refs={}
    contract=read(out/'random_contract.json')
    def check():
        if time.time()>p['delivery_epoch']-30:raise TimeoutError('evaluation deadline')
    graphs={}
    numeric=['nodes','edges','edge_abs_error','degree_W1','joint_TV','triangles','triangle_abs_error','triangle_composition_TV','MP2','MP2_abs_error','MP2_TV','MP3','MP3_abs_error','MP3_TV','LCC','components','isolates','four_cycles','edge_jaccard','type_coverage']
    for ds,parent in p['models'].items():
        check();z=np.load(parent['data']);types,schema,edges=z['types'],z['schema'],z['edges'];edges=edges.T if edges.shape[0]!=2 else edges
        names=['Director','Actor','Movie','Genre'] if ds=='IMDB' else ['Gene','Disease','Chemical','Species']
        ref=ev.M.make_graph(types,edges,schema);rs=ev.M.stats(ref,ev.F,names,check);refs[ds]=rs
        case=sc.w.common.make_case(types,edges,tuple(map(tuple,schema)))
        ids,fam,query=sc.w.common.plan_for(case,2026111402,1).block(0)
        scorer=sc.Score(ds,case,query,fam,np.isin(ids,case['clean_ids']).astype(np.uint8))
        for job in [x for x in p['all_jobs'] if x['dataset']==ds]:
            check();directory=Path(job['directory']);ss=job['sample_seed'];r=job['rounds'];key=(ds,r,ss)
            if not (directory/'summary.json').exists():missing.append(dict(job=job['id'],reason='no completed chain'));continue
            sm=read(directory/'summary.json');ident=read(directory/'identity.json')
            assert sm['status']=='complete' and sm['steps']==1000 and ev.sha(directory/'graph.npz')==sm['graph_sha256']
            assert ident['T']==1000 and ident['R']==r and ident['sample_seed']==ss and ident['model_hash']==parent['model_digest']
            assert ident['updates']==2 and ident['train_seed']==parent['train_seed']
            steps=[json.loads(x) for x in (directory/'steps.jsonl').read_text().splitlines()]
            assert [x['time'] for x in steps]==list(range(1000,0,-1))
            expected=contract[ds][str(ss)]
            assert ident['initial_hash']==expected['initial_hash']
            assert [x['uniform_hash'] for x in steps]==expected['step_uniform_hashes']
            assert all(x['exact_coverage'] and x['covered_slots']==len(ids) and x['blocks']==r for x in steps)
            verification.append(dict(job=job['id'],reused=job['reused'],complete_1000=True,matched_initial_and_full_uniforms=True,model_identity=True,coverage=True))
            graph=np.load(directory/'graph.npz');g=ev.M.make_graph(graph['types'],graph['edges'],graph['schema'])
            assert np.array_equal(g['types'],types) and np.array_equal(g['schema'],schema)
            graphs[key]=g
            fullpath=cache/(job['id']+'.json')
            if fullpath.exists():full=read(fullpath);assert full['graph_sha256']==sm['graph_sha256']
            else:
                st=ev.M.stats(g,ev.F,names,check);cmp=ev.M.compare(g,st,ref,rs,ev.F,names)
                lg,mapping=ev.M.lcc(g);ls=ev.M.stats(lg,ev.F,names,check);lc=ev.M.compare(lg,ls,ref,rs,ev.F,names)
                bits=np.zeros(case['space'].total,dtype=np.uint8);ff,ll=case['space'].encode(g['edges']);bits[case['space'].offsets[ff]+ll]=1
                terms=scorer(bits[ids]);full=dict(graph_sha256=sm['graph_sha256'],reference=rs,raw=dict(statistics=st,comparison=cmp),
                    LCC=dict(statistics=ls,comparison=lc,mapping=mapping),objective=dict(joint=terms[0],count=terms[1],total=sum(terms[:2]),by_type=terms[2]))
                ev.save(fullpath,full)
            costs.append(dict(dataset=ds,rounds=r,sample_seed=ss,**full['objective']))
            for view in ('raw','LCC'):
                st=full[view]['statistics'];cmp=full[view]['comparison']
                rows.append(dict(dataset=ds,rounds=r,train_seed=parent['train_seed'],sample_seed=ss,view=view,
                    nodes=st['nodes'],edges=st['edges'],edge_abs_error=abs(st['edges']-rs['edges']),reference_edges=rs['edges'],
                    degree_W1=cmp['main_incident4_W1'],joint_TV=cmp['main_joint_degree_TV'],triangles=st['triangles'],reference_triangles=rs['triangles'],
                    triangle_abs_error=abs(st['triangles']-rs['triangles']),triangle_composition_TV=cmp['triangle_composition_TV'],
                    MP2=st['mp'][1]['total'],MP2_abs_error=abs(st['mp'][1]['total']-rs['mp'][1]['total']),MP2_TV=cmp['MP_composition_TV'][1],
                    MP3=st['mp'][2]['total'],MP3_abs_error=abs(st['mp'][2]['total']-rs['mp'][2]['total']),MP3_TV=cmp['MP_composition_TV'][2],
                    LCC=st['lcc'],components=st['components'],isolates=st['isolates'],four_cycles=st['four_cycles'],edge_jaccard=cmp['EO_jaccard'],type_coverage=len(names)-len(cmp['missing_types'])))
                all_metrics.extend(dict(dataset=ds,rounds=r,sample_seed=ss,view=view,metric=k,value=v) for k,v in ev.flat(full[view]).items())
        for r in p['rounds']:
            for a,b in itertools.combinations(p['sample_seeds'],2):
                if (ds,r,a) not in graphs or (ds,r,b) not in graphs:continue
                ea=set(map(tuple,graphs[(ds,r,a)]['edges'].T));eb=set(map(tuple,graphs[(ds,r,b)]['edges'].T))
                pairs.append(dict(dataset=ds,rounds=r,seed1=a,seed2=b,jaccard=len(ea&eb)/len(ea|eb) if ea|eb else None,identical=ea==eb))
        for ss in p['sample_seeds']:
            for ra,rb in [(1,10),(1,20),(10,20)]:
                ja=next(j for j in p['all_jobs'] if j['dataset']==ds and j['rounds']==ra and j['sample_seed']==ss)
                jb=next(j for j in p['all_jobs'] if j['dataset']==ds and j['rounds']==rb and j['sample_seed']==ss)
                for t in p['snapshot_times']:
                    pa=Path(ja['directory'])/f'Xt_{t:04d}.npz';pb=Path(jb['directory'])/f'Xt_{t:04d}.npz'
                    if pa.exists() and pb.exists():
                        a=np.load(pa)['xt'];b=np.load(pb)['xt'];trajectories.append(dict(dataset=ds,sample_seed=ss,rounds_a=ra,rounds_b=rb,t=t,edge_disagreements=int((a!=b).sum()),edges_a=int(a.sum()),edges_b=int(b.sum())))
    summaries=[];deltas=[]
    for ds in p['models']:
        for view in ('raw','LCC'):
            for r in p['rounds']:
                rr=[x for x in rows if x['dataset']==ds and x['view']==view and x['rounds']==r]
                if not rr:continue
                row=dict(dataset=ds,rounds=r,view=view,training_models=1,samples=len(rr),complete_panel=len(rr)==4)
                for k in numeric:
                    vals=[x[k] for x in rr if x[k] is not None]
                    row[k+'_mean']=float(np.mean(vals)) if vals else None;row[k+'_sample_sd']=float(np.std(vals,ddof=1)) if len(vals)>1 else None
                summaries.append(row)
                if r!=1:
                    for x in rr:
                        b=next((z for z in rows if z['dataset']==ds and z['rounds']==1 and z['view']==view and z['sample_seed']==x['sample_seed']),None)
                        if b:deltas.append(dict(dataset=ds,rounds=r,view=view,sample_seed=x['sample_seed'],**{k:x[k]-b[k] if x[k] is not None and b[k] is not None else None for k in numeric}))
    for name,table in [('per_graph',rows),('all_metrics',all_metrics),('objective_scores',costs),('model_summary',summaries),('paired_differences',deltas),('sample_pairs',pairs),('trajectory_differences',trajectories)]:
        ev.save(out/(name+'.json'),table);ev.csvsave(out/(name+'.csv'),table)
    ev.save(out/'reference_metrics.json',refs)
    for path,h in p['bindings'].items():assert ev.sha(path)==h,path
    ev.save(out/'verification.json',dict(checks=verification,missing=missing,completed_graphs=len(rows)//2,expected_graphs=24,
        all_bound_assets_unchanged=True,training_models_per_dataset=1,sample_seed_not_training_replicate=True,
        query_hash_not_compared_across_R='Query sets per block intentionally differ; full support covered and full-slot sampling uniforms match'))
    ev.save(out/'evaluation_receipt.json',dict(seconds=time.time()-start,GPU=0,missing=missing))
    print(json.dumps(dict(graphs=len(rows)//2,missing=missing,seconds=time.time()-start)))
if __name__=='__main__':main()
