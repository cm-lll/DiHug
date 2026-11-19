"""CPU-only whole-graph/LCC evaluation, paired seed tables and diversity audit."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import sys,json,time,csv,itertools,statistics,traceback,hashlib,resource
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'research_branches/joint_transport_ablation_20261113'))
from reliability_analysis import M,F,inc,organization_metrics
import numpy as np
from effect_hard_score import score as hard_score
def write(p,x):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False));tmp.replace(p)
def table(p,rows):
    write(p.with_suffix('.json'),rows)
    if rows:
        with p.with_suffix('.csv').open('w') as f:
            w=csv.DictWriter(f,list(dict.fromkeys(k for row in rows for k in row)));w.writeheader();w.writerows(rows)
def main():
    out=Path(sys.argv[1]);start=time.time();cpu=time.process_time();p=json.loads((out/'protocol.json').read_text())
    ev=out/'evaluation';ev.mkdir(exist_ok=True);refs={};contexts={};rows=[];relations=[];types=[];missing=[];failed=[];graphsets={}
    def check():
        if time.time()>p['delivery_epoch']-10:raise TimeoutError('delivery boundary')
    bindings=json.loads((out/'asset_recheck.json').read_text())['bindings']
    for path,h in bindings.items():
        check();assert inc.assets.sha(path)==h,path
    for row in p['parents']:
        ds=row['dataset'];names=row['config']['datasets'][ds]['type_names']
        if ds in refs:continue
        z=np.load(row['data']);edges=z['edges'];edges=edges if edges.shape[0]==2 else edges.T
        g=M.make_graph(z['types'],edges,z['schema']);rs=M.stats(g,F,names,check)
        refs[ds]=(g,rs,names)
        contexts[ds]=inc.assets.load(ds,dict(row,readout=row['parent_readout']))
    write(out/'reference_metrics.json',{d:v[1] for d,v in refs.items()})
    inventory=[]
    for r in p['reused']:
        inventory.append(dict(model=r['model'],dataset=r['dataset'],train_seed=r['train_seed'],arm='parent',readout='parent_EMA3000',sample_seed=r['sample_seed'],
            directory=r['path'],graph=r['graph'],reused=True,old_evaluation=r['evaluation']))
    for t in p['tasks']:
        d=out/t['model']/t['arm']/t['readout']/f'{t["arm"]}_seed{t["sample_seed"]}'
        inventory.append(dict(model=t['model'],dataset=t['dataset'],train_seed=t['train_seed'],arm=t['arm'],readout=t['readout'],sample_seed=t['sample_seed'],
            directory=str(d),graph=str(d/'graph.npz'),reused=False))
    write(out/'inventory.json',inventory)
    def flatten(item,full):
        ref,rs,names=refs[item['dataset']];org=full['organization']
        ident={k:item[k] for k in ('model','dataset','train_seed','arm','readout','sample_seed','reused','graph')}
        for view in ('raw','LCC'):
            st=full[view]['statistics'];co=full[view]['comparison']
            rows.append(dict(**ident,view='whole' if view=='raw' else 'LCC',nodes=st['nodes'],edges=st['edges'],reference_edges=rs['edges'],
                edge_abs_error=abs(st['edges']-rs['edges']),degree_W1=co['main_incident_W1'],joint_TV=co['main_joint_degree_TV'],
                typed_closure_distribution=full['new_descriptor']['closure'] if view=='raw' else None,
                typed_propagation_distribution=full['new_descriptor']['propagation'] if view=='raw' else None,
                triangles=st['triangles'],reference_triangles=rs['triangles'],triangle_abs_error=abs(st['triangles']-rs['triangles']),triangle_composition_TV=co['triangle_composition_TV'],
                MP2=st['mp'][1]['total'],MP2_abs_error=abs(st['mp'][1]['total']-rs['mp'][1]['total']),MP2_TV=co['MP_composition_TV'][1],
                MP3=st['mp'][2]['total'],MP3_abs_error=abs(st['mp'][2]['total']-rs['mp'][2]['total']),MP3_TV=co['MP_composition_TV'][2],
                LCC=st['lcc'],components=st['components'],isolates=st['isolates'],maximum_degree=st['maximum_degree'],
                reachable_pair_fraction=st['reachable_pair_fraction'],four_cycles=st['four_cycles'],four_cycle_composition_TV=co['four_cycle_composition_TV'],
                joint_relation_degree_L1=org['joint_relation_degree_L1'] if view=='raw' else None,
                low_fraction_error=org['type_low_fraction_error'] if view=='raw' else None,
                hub_top10_mean_error=org['type_top10_degree_mean_error'] if view=='raw' else None,
                reference_edge_jaccard=co['EO_jaccard'],reference_overlap=co['edge_overlap_count']))
            for ri,(a,b) in enumerate(ref['schema']):
                relations.append(dict(**ident,view=view,relation=ri,endpoints=f'{names[int(a)]}--{names[int(b)]}',
                    edges=st['family_edges'][ri],reference_edges=rs['family_edges'][ri],
                    edge_abs_error=abs(st['family_edges'][ri]-rs['family_edges'][ri]),degree_W1=co['family_endpoint_W1'][str(ri)]))
            for name in names:
                types.append(dict(**ident,view=view,type=name,statistics=st['typed_degree'].get(name),
                    degree_W1=co['type_total_degree_W1'].get(name),joint_TV=co['joint_TV_by_type'].get(name)))
    for item in inventory:
        check();d=Path(item['directory']);key=f'{item["model"]}_{item["arm"]}_{item["readout"]}_{item["sample_seed"]}'
        if not (d/'summary.json').exists():missing.append(item);continue
        try:
            summary=json.loads((d/'summary.json').read_text());assert summary['status']=='complete' and summary['steps']==1000
            assert inc.assets.sha(item['graph'])==summary['graph_sha256']
            ref,rs,names=refs[item['dataset']];z=np.load(item['graph']);g=M.make_graph(z['types'],z['edges'],z['schema'])
            assert np.array_equal(g['types'],ref['types']) and np.array_equal(g['schema'],ref['schema'])
            if item['reused']:
                full=json.loads(Path(item['old_evaluation']).read_text());assert full['graph_sha256']==summary['graph_sha256']
            else:
                assert summary['paired_initial_query_uniform'] and summary['checkpoint_unchanged']
                sc=contexts[item['dataset']];bits=inc.assets.bits(sc,z)
                org=organization_metrics(sc,bits,g['degree'],{'LCC':rs['lcc']})
                st=M.stats(g,F,names,check);co=M.compare(g,st,ref,rs,F,names)
                lc,mapping=M.lcc(g);ls=M.stats(lc,F,names,check);lco=M.compare(lc,ls,ref,rs,F,names)
                full=dict(identity=item,graph_sha256=summary['graph_sha256'],organization=org,
                    raw=dict(statistics=st,comparison=co),LCC=dict(statistics=ls,comparison=lco,mapping=mapping))
            full['new_descriptor']=hard_score(item['graph'],item['dataset'])
            write(ev/(key+'.json'),full);flatten(item,full)
            graphsets.setdefault((item['model'],item['arm'],item['readout']),[]).append((item['sample_seed'],set(map(tuple,g['edges'].T))))
        except Exception:failed.append(dict(**item,traceback=traceback.format_exc()))
    table(out/'per_graph',rows);table(out/'per_relation',relations);table(out/'per_type',types)
    write(out/'missing.json',missing);write(out/'evaluation_failures.json',failed)
    diversity=[]
    for (model,arm,readout),graphs in graphsets.items():
        for (s1,a),(s2,b) in itertools.combinations(graphs,2):
            common=len(a&b);union=len(a|b)
            diversity.append(dict(model=model,arm=arm,readout=readout,seed1=s1,seed2=s2,
                edges1=len(a),edges2=len(b),common=common,jaccard=common/union if union else None,
                fraction1=common/len(a) if a else None,fraction2=common/len(b) if b else None,exact_edge_duplicate=a==b))
    table(out/'sample_pairs',diversity)
    metadata=('model','dataset','train_seed','arm','readout','sample_seed','reused','graph','view')
    metrics=[k for k in rows[0] if k not in metadata] if rows else []
    groups={}
    for r in rows:groups.setdefault(tuple(r[k] for k in ('model','dataset','train_seed','arm','readout','view')),[]).append(r)
    means=[]
    for key,rr in groups.items():
        info=dict(zip(('model','dataset','train_seed','arm','readout','view'),key));info['samples']=len(rr)
        for metric in metrics:
            values=[r[metric] for r in rr if r[metric] is not None]
            info[metric+'_mean']=statistics.mean(values) if values else None
            info[metric+'_sample_SD']=statistics.stdev(values) if len(values)>1 else None
            info[metric+'_defined_n']=len(values)
        means.append(info)
    table(out/'model_summary',means)
    paired=[]
    for r in means:
        if r['arm']=='parent':continue
        for target in ('parent','CE'):
            if target=='CE' and r['arm']!='calibrated_micro_macro':continue
            base=next((x for x in means if x['model']==r['model'] and x['view']==r['view'] and x['arm']==target and
                       (target=='parent' or x['readout']==r['readout'])),None)
            if base is None:continue
            info={k:r[k] for k in ('model','dataset','train_seed','arm','readout','view','samples')};info['against']=target
            info['complete_pair']=r['samples']==base['samples']==4
            for metric in metrics:
                a=r[metric+'_mean'];b=base[metric+'_mean'];info[metric+'_delta']=a-b if a is not None and b is not None else None
            paired.append(info)
    table(out/'paired_differences',paired)
    cross=[]
    for key in sorted(set((r['dataset'],r['arm'],r['readout'],r['view']) for r in means)):
        rr=[r for r in means if (r['dataset'],r['arm'],r['readout'],r['view'])==key and r['samples']==4]
        z=dict(zip(('dataset','arm','readout','view'),key));z['complete_training_models']=len(rr)
        for m in metrics:
            vals=[r[m+'_mean'] for r in rr if r[m+'_mean'] is not None]
            z[m+'_mean']=statistics.mean(vals) if vals else None;z[m+'_training_seed_SD']=statistics.stdev(vals) if len(vals)>1 else None
        cross.append(z)
    table(out/'dataset_summary',cross)
    release=json.loads((out/'release.json').read_text())
    complete=len(rows)==96 and not failed and not missing
    receipt=dict(status='complete' if complete else 'partial',unique_graphs=len(rows)//2,new_graphs=sum(not r['reused'] for r in rows)//2,
        reused_graphs=sum(r['reused'] for r in rows)//2,missing=len(missing),failures=len(failed),
        CPU_seconds=time.process_time()-cpu,analysis_wall_seconds=time.time()-start,
        total_wall_minutes=(time.time()-p['start_epoch'])/60,GPU_hours=release['GPU_hours'],
        all_owned_GPU_processes_exited=release['all_owned_processes_exited'],new_training_updates=sum(json.loads(f.read_text())['updates'] for f in out.glob('*_transport/*/completion.json')),
        peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,epoch=time.time())
    write(out/'completion.json',receipt)
    text=['# 固定128步EMA终点匹配结构效果','',f'状态：{receipt["status"]}；{receipt["new_graphs"]}/32新链、{receipt["reused_graphs"]}/16旧图；{receipt["GPU_hours"]:.4f} GPU小时；总墙钟{receipt["total_wall_minutes"]:.1f}分钟。',
        '','原图为主；表中每行先平均同一模型四次采样。完整采样标准差见model_summary，跨训练seed标准差见dataset_summary；不得把四张图当四个训练重复。',
        '','|数据/训练seed|目标|权重|n|边数|度W1|联合TV|联合度L1|三角数/误差|MP3误差|LCC|低度误差|hub误差|',
        '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    def num(x):return '缺' if x is None else f'{x:.4g}'
    for r in means:
        if r['view']!='whole':continue
        text.append('|'+ '|'.join([f'{r["dataset"]}/{r["train_seed"]}',r['arm'],r['readout'],str(r['samples']),
            *[num(r[m+'_mean']) for m in ('edges','degree_W1','joint_TV','joint_relation_degree_L1')],
            num(r['triangles_mean'])+'/'+num(r['triangle_abs_error_mean']),
            *[num(r[m+'_mean']) for m in ('MP3_abs_error','LCC','low_fraction_error','hub_top10_mean_error')]])+'|')
    text.extend(['','## 完整配对差异：负值表示误差减少','',
        '|训练seed|权重|比较|度W1差|联合度L1差|三角绝对误差差|MP3误差差|低度差|hub差|LCC差|',
        '|---|---|---|---|---|---|---|---|---|---|'])
    for r in paired:
        if r['view']!='whole' or not r['complete_pair']:continue
        text.append('|'+ '|'.join([str(r['train_seed']),r['readout'],r['arm']+'−'+r['against'],
            *[num(r[m+'_delta']) for m in ('degree_W1','joint_relation_degree_L1','triangle_abs_error','MP3_abs_error','low_fraction_error','hub_top10_mean_error','LCC')]])+'|')
    text.extend(['','LCC越大不自动代表更接近参考；参考图本身可能含多个分量。边数、三角/MP数量与组成、逐关系、原图/LCC和全部样本见CSV/JSON。',
        '','四训练seed、每数据集两seed；历史采样seed属于开发面板。只评价预定EMA生成，不挑raw/EMA。128步以父EMA3000为初态、新Adam、IID单时间配对更新；不是从头稳定性验证。',
        '','sample_pairs保留每模型六对编号边重合和重复检查；未做精确同构，不能排除重编号复制，也不能用低重合证明高质量多样性。',
        '','GPU序列化状态/随机合同核验与CUDA重算逐位一致是不同主张，后者仍未认证。本票有配对128步更新；没有超出面板的优化或自动后续实验。',
        '',f'失败{len(failed)}，缺项{len(missing)}；见evaluation_failures.json/missing.json。资源见release.json/allocations.json。'])
    (out/'REPORT_ZH.md').write_text('\n'.join(text)+'\n')
    write(out/'handoff.json',dict(status=receipt['status'],next='Read complete paired tables and all structural costs; no automatic extension or next experiment.',report=str(out/'REPORT_ZH.md')))
    print(json.dumps(receipt))
if __name__=='__main__':main()
