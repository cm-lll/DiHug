"""CPU full original/LCC evaluation, four samples nested within each model."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,time,importlib.util,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/single_step_formal_resume_20261111'
s=importlib.util.spec_from_file_location('metrics_old',ROOT/'research_branches/terminal_structure_real_20261109/evaluate.py');lib=importlib.util.module_from_spec(s);s.loader.exec_module(lib)
M,F,np=lib.M,lib.F,lib.np
def check():
    if time.time()>json.loads((OUT/'window.json').read_text())['delivery_stop_epoch']-30:raise TimeoutError('delivery deadline')
def main():
    idx=int(sys.argv[1]);proto=json.loads((OUT/'protocol.json').read_text());task=proto['tasks'][idx];row=task['parent'];ds=row['dataset'];dest=OUT/task['id'];start=time.time();result=[];errors=[];allscalar=[]
    data=np.load(row['data']);ed=data['edges'];ed=ed.T if ed.shape[0]!=2 else ed;types=data['types'];schema=data['schema'];names=['Director','Actor','Movie','Genre'] if ds=='IMDB' else ['Gene','Disease','Chemical','Species'];ref=M.make_graph(types,ed,schema);rs=M.stats(ref,F,names,check)
    sources=[(task['id'],dest,False)]
    if task['B']==1:sources.append(('ordinary_CE_background',Path(row['checkpoint']).parent/'ema_0.99',True))
    for arm,base,old in sources:
        for ss in proto['sample_seeds']:
            folder=base/f'sample_{ss}';gp=folder/'graph.npz'
            try:
                check();binding=json.loads((folder/'binding.json').read_text());assert binding['graph_sha256']==lib.sha(gp)
                if old:
                    assert binding['train_seed']==row['seed'] and binding['updates']==3000 and binding['sample_seed']==ss and binding['configuration']=='A_late'
                    assert lib.sha(Path(row['checkpoint']).parent/'readout_ema_0.99.pt')==binding['model_sha256']
                else:assert binding['readout_sha256']==lib.sha(dest/'EMA3000.pt') and binding['updates']==3000
                steps=[json.loads(v) for v in (folder/'steps.jsonl').read_text().splitlines()];contract=json.loads(Path(proto['sampling_contracts']).read_text())[ds+'_'+str(ss)]['steps'];assert len(steps)==1000
                for a,c in zip(steps,contract):assert a['uniform_sha256']==c['uniform_sha256'] and a['query_plan_sha256']==c['query_plan_sha256'] and a['exact_coverage']
                z=np.load(gp);gg=M.make_graph(z['types'],z['edges'],z['schema']);assert np.array_equal(gg['types'],types) and np.array_equal(gg['schema'],schema);lg,mapping=M.lcc(gg);full={}
                for view,graph in [('original',gg),('LCC',lg)]:
                    st=M.stats(graph,F,names,check);cmp=M.compare(graph,st,ref,rs,F,names);full[view]=dict(statistics=st,comparison=cmp,reference=rs)
                    ident=dict(dataset=ds,train_seed=row['seed'],arm=arm,B=None if old else task['B'],K=None if old else task['K'],sample_seed=ss,view=view)
                    result.append(dict(**ident,nodes=st['nodes'],edges=st['edges'],reference_edges=rs['edges'],edge_error=abs(st['edges']-rs['edges']),degree_W1=cmp['main_incident4_W1'],joint_TV=cmp['main_joint_degree_TV'],triangles=st['triangles'],reference_triangles=rs['triangles'],triangle_error=abs(st['triangles']-rs['triangles']),triangle_composition_TV=cmp['triangle_composition_TV'],MP2_error=abs(st['mp'][1]['total']-rs['mp'][1]['total']),MP3_error=abs(st['mp'][2]['total']-rs['mp'][2]['total']),MP2_TV=cmp['MP_composition_TV'][1],MP3_TV=cmp['MP_composition_TV'][2],lcc=st['lcc'],components=st['components'],isolates=st['isolates']))
                    allscalar.extend(dict(**ident,metric=k,value=v) for k,v in lib.flat(full[view]).items())
                lib.save(dest/f'metrics_{arm}_{ss}.json',dict(source=str(gp),source_sha256=lib.sha(gp),metrics=full,LCC_mapping=mapping))
            except BaseException as ex:errors.append(dict(arm=arm,sample_seed=ss,error=repr(ex),traceback=traceback.format_exc()))
    lib.save(dest/'per_graph.json',result);lib.csvsave(dest/'per_graph.csv',result);lib.save(dest/'all_metrics.json',allscalar);lib.csvsave(dest/'all_metrics.csv',allscalar);lib.save(dest/'evaluation_receipt.json',dict(seconds=time.time()-start,graphs=len(result)//2,errors=errors,GPU=0));print(json.dumps(dict(task=task['id'],graphs=len(result)//2,errors=errors)))
if __name__=='__main__':main()
