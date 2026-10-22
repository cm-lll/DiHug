"""CPU-only re-evaluation of archived graphs; never trains or samples a graph.

Original metric source is byte-for-byte preserved. Namespace packages avoid
executing unrelated production-model re-exports in the old __init__.py.
"""
import argparse,csv,hashlib,json,os,sys,time,types,resource,platform
from pathlib import Path

ROOT=Path(__file__).resolve().parent
ATOL=1e-10
METRICS=('edges','lcc','triangles','clustering','assortativity','degree_mmd','typed_triangle_tv','MP2','MP3','edge_overlap_percent')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--max-seconds',type=float,default=300);parser.add_argument('--metric-version',choices=('submission-git','retained-working-tree'),default='submission-git')
    args=parser.parse_args();start=time.perf_counter();deadline=start+min(300,args.max_seconds)
    out=args.output.resolve()
    if out==ROOT or ROOT in out.parents or out in ROOT.parents:raise ValueError('output overlaps protected archived inputs/source')
    if out.exists():raise FileExistsError('new output required')
    out.mkdir(parents=True);status='failed';error=None;groups=[];maximum=0.;checks=[]
    def check():
        if (out/'STOP').exists():raise InterruptedError('STOP')
        if time.perf_counter()>=deadline:raise TimeoutError('bounded CPU audit incomplete')
    try:
        check();inventory=read(ROOT/'source_inventory.json')
        for row in inventory:
            check()
            if sha(ROOT/row['path'])!=row['sha256']:raise ValueError('archived input hash mismatch '+row['path'])
        # No imports from mutable installed DiHug or another experiment.
        sys.dont_write_bytecode=True
        for name,folder in (('sparse_diffusion','sparse_diffusion'),('sparse_diffusion.structure_energy','sparse_diffusion/structure_energy'),('scripts','scripts')):
            module=types.ModuleType(name);module.__path__=[str(ROOT/'original_source'/folder)];sys.modules[name]=module
        if args.metric_version=='submission-git':sys.modules['sparse_diffusion.structure_energy'].__path__.insert(0,str(ROOT/'original_git_source/sparse_diffusion/structure_energy'))
        import numpy as np,torch,importlib.metadata as metadata
        from sparse_diffusion.structure_energy.graph import load_typed_graph
        from sparse_diffusion.structure_energy.heterogeneous_metrics import compare_heterogeneous_graphs,degree_mmd_report
        from scripts.evaluate_heterogeneous_generation import load_graph
        from scripts.evaluate_external_graphs_canonical import load_hgen_reference,load_external,type_pair_graph,simple_metrics
        torch.set_num_threads(1)
        env={'python':platform.python_version(),'packages':{n:metadata.version(n) for n in ('numpy','scipy','torch','networkx')},'threads':1,'CUDA_initialized':torch.cuda.is_initialized(),'ATOL':ATOL,'RTOL':0,'model_calls':0,'new_graphs':0,'metric_version':args.metric_version}
        write(out/'environment.json',env)
        table={}
        with (ROOT/'original_reports/paper/results_20260727/fullgraph_core_8run.tsv').open() as f:
            for row in csv.DictReader(f,delimiter='\t'):table[(row['dataset'],row['method'])]=row
        with (ROOT/'original_reports/paper/results_20260727/fullgraph_distribution_8run.tsv').open() as f:
            for row in csv.DictReader(f,delimiter='\t'):table[(row['dataset'],row['method'])].update(row)
        cases={};mapping={}
        for dataset in ('imdb','pubmed'):
            check();target=load_typed_graph(ROOT/'reference'/dataset,active_only=True);cases[dataset]=target
            he,ht=load_hgen_reference(ROOT/'hgen_reference'/dataset)
            if not np.array_equal(ht,target.node_types) or he!=set(target.edge_family):raise ValueError('HGEN node/reference alignment not reproduced')
            raw=torch.load(ROOT/'reference'/dataset/'nodes.pt',map_location='cpu',weights_only=False)
            dense_to_raw=[int(i) for name in target.type_names for i in raw[name]['ids']]
            mapping[dataset]={'dense_to_original_node_id':dense_to_raw,'node_types':ht.tolist(),'type_names':list(target.type_names),
                'families':{k:{'src_type':s.src_type,'dst_type':s.dst_type,'edges':len(target.family_edges[k])} for k,s in target.family_specs.items()},
                'HGEN_alignment_recomputed':True,'HGEN_types_origin':'deterministically recovered from archived new_node_dict.p and adj_matrix.p LCC ordering; NOT originally stored in graph exports'}
            write(out/(dataset+'_reference_graph.json'),target.to_json_dict())
        write(out/'node_and_relation_mapping.json',mapping)
        def compare_numbers(actual,expected,label):
            nonlocal maximum
            err=abs(float(actual)-float(expected));maximum=max(maximum,err)
            if err>ATOL:raise ValueError(f'{label} differs: {actual} vs {expected}, error={err}, atol={ATOL}')
        for group in read(ROOT/'asset_index.json')['groups']:
            check();tick=time.perf_counter();d=group['dataset'];method=group['method'];target=cases[d];old=read(ROOT/group['original_report']);rows=[];errors=[]
            print('START',d,method,flush=True)
            for i,record in enumerate(group['graphs']):
                check();path=ROOT/record['path'];historical=old['rows'][i]
                if method=='HGEN':
                    edges,types0=load_external(path,target.num_nodes)
                    node_types=target.node_types if types0 is None else types0
                    generated=type_pair_graph(node_types,target.type_names,edges)
                    typed_target=type_pair_graph(target.node_types,target.type_names,set(target.edge_family))
                    result=compare_heterogeneous_graphs(typed_target,generated)
                    values=simple_metrics(target.num_nodes,edges)
                    values.update(degree_mmd_report(target.degrees(),generated.degrees()))
                    for k in ('edges','lcc','triangles','clustering','assortativity','degree_mmd'):compare_numbers(values[k],historical[k],d+'/'+method+'/'+str(i)+'/'+k)
                else:
                    generated=load_graph(path);result=compare_heterogeneous_graphs(target,generated)
                    values=result['global']['generated'].copy();values['degree_mmd']=result['global']['degree_mmd']
                    for k in ('edges','lcc','triangles','clustering','assortativity'):compare_numbers(values[k],historical['global']['generated'][k],d+'/'+method+'/'+str(i)+'/'+k)
                    compare_numbers(values['degree_mmd'],historical['global']['degree_mmd'],d+'/'+method+'/'+str(i)+'/MMD')
                    for k in ('2','3'):compare_numbers(result['typed_metapaths'][k]['tv'],historical['typed_metapaths'][k]['tv'],d+'/'+method+'/'+str(i)+'/MP'+k)
                    compare_numbers(result['typed_triangles']['tv'],historical['typed_triangles']['tv'],d+'/'+method+'/'+str(i)+'/TriTV')
                values.update(typed_triangle_tv=result['typed_triangles']['tv'],MP2=result['typed_metapaths']['2']['tv'],MP3=result['typed_metapaths']['3']['tv'],edge_overlap_percent=result['edge_identity_diagnostic']['edge_overlap_percent'])
                expected_eo=historical['edge_overlap_percent'] if method=='HGEN' else historical['edge_identity_diagnostic']['edge_overlap_percent']
                compare_numbers(values['edge_overlap_percent'],expected_eo,d+'/'+method+'/'+str(i)+'/EO')
                rows.append(dict(sample=record['sample'],graph_sha256=record['sha256'],metrics=values))
                write(out/f'{d}_{method}_sample{i:03d}_metrics.json',result)
                check()
            mean={k:float(np.mean([v['metrics'][k] for v in rows])) for k in METRICS}
            std={k:float(np.std([v['metrics'][k] for v in rows],ddof=0)) for k in METRICS}
            expected=table[(d,method)]
            for k in ('edges','lcc','triangles','clustering','assortativity'):
                compare_numbers(mean[k],expected[k+'_mean'],d+'/'+method+'/table/'+k)
                compare_numbers(std[k],expected[k+'_std'],d+'/'+method+'/table/std/'+k)
            for k,column in (('degree_mmd','degree_mmd_mean'),('typed_triangle_tv','typed_triangle_tv_mean'),('MP2','metapath_l2_tv_mean'),('MP3','metapath_l3_tv_mean'),('edge_overlap_percent','edge_overlap_percent_mean')):
                compare_numbers(mean[k],expected[column],d+'/'+method+'/table/'+column)
            item={'dataset':d,'method':method,'n':len(rows),'rows':rows,'mean':mean,'population_std':std,'table2_all_metrics_reproduced':True,'seconds':time.perf_counter()-tick}
            write(out/f'{d}_{method}_summary.json',item);groups.append(item)
            print('END',d,method,'passed',item['seconds'],flush=True)
        # Recheck source bytes, including all eighty originals' archive copies.
        for row in inventory:
            check()
            if sha(ROOT/row['path'])!=row['sha256']:raise ValueError('input changed during audit')
        if torch.cuda.is_initialized():raise ValueError('unexpected GPU initialization')
        check();status='passed'
    except (TimeoutError,InterruptedError) as exc:status='incomplete';error=repr(exc);raise
    except BaseException as exc:error=repr(exc);raise
    finally:
        write(out/'summary.json',{'status':status,'error':error,'completed_groups':len(groups),'planned_groups':10,'completed_graphs':sum(g['n'] for g in groups),'planned_graphs':80,
            'groups':groups,'maximum_numerical_error':maximum,'ATOL':ATOL,'RTOL':0,'wall_seconds':time.perf_counter()-start,'peak_rss_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            'model_calls':0,'training_updates':0,'new_graphs':0,'original_results_modified':False,'metric_version':args.metric_version,'expectation':'eight means and population scalar standard deviations; no graph selection'})

if __name__=='__main__':main()
