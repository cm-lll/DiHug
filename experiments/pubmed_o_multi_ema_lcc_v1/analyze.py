"""All64 raw/LCC CPU analysis, fixed source/seed/readout identity, no models."""
import argparse,collections,csv,datetime,hashlib,itertools,json,math,pathlib,resource,shutil,statistics,time
import numpy as np
import metrics as M
Counter=collections.Counter
SEEDS=[2026102812,2026102833,2026102841,2026102842,2026102843,2026102844,2026102845,2026102846]
READOUTS=['raw','ema_0.9','ema_0.97','ema_0.99'];SAMPLES=[2026102821,2026102822];NAMES=['Gene','Disease','Chemical','Species']
def save(p,x):
    p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    def convert(v):
        if isinstance(v,np.ndarray):return v.tolist()
        if isinstance(v,np.generic):return v.item()
        raise TypeError(type(v).__name__)
    p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False,default=convert))
def csv_save(p,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def mean(values):return sum(values)/len(values) if values and all(v is not None and math.isfinite(float(v)) for v in values) else None
def cohort(seed):return 'old_two' if seed in SEEDS[:2] else 'new_six'
def difference(a,b):return None if a is None or b is None else a-b

def scalar(s,ref,comp):
    raw={k:v for k,v in s.items() if (v is None or isinstance(v,(int,float,np.integer,np.floating))) and k not in ['HGEN_native_powerlaw']}
    raw['HGEN_native_powerlaw']=None
    for t,name in enumerate(NAMES):
        d=s['typed_degree'][name];rd=ref['typed_degree'][name];n=d['count'];threshold=max(1,int(math.ceil(rd['p99'])))
        for key in ['count','maximum','second','mean','zero','low_le1','p90','p99','gini_standard','top5_degree_share']:raw['type_'+name+'_'+key]=d[key]
        raw['type_'+name+'_share']=n/s['nodes'];raw['type_'+name+'_zero_fraction']=d['zero']/n if n else None;raw['type_'+name+'_low_le1_fraction']=d['low_le1']/n if n else None
        raw['type_'+name+'_fixed_ref_P99_ge_count']=sum(v>=threshold for v in d['sorted_degrees']);raw['type_'+name+'_fixed_ref_P99_threshold']=threshold
        for r,col in enumerate(d['by_relation']):
            for key in ['mean','max','zero','low_le1','p90','p99']:raw['type_'+name+'_relation'+str(r)+'_'+key]=col[key]
    for r,(edges,cap) in enumerate(zip(s['family_edges'],s['capacities'])):raw['family'+str(r)+'_edges']=edges;raw['family'+str(r)+'_capacity']=cap;raw['family'+str(r)+'_density']=edges/cap if cap else None
    for pattern in itertools.combinations_with_replacement(range(4),3):raw['triangle_'+str(pattern)]=s['typed_triangles'].get(str(pattern),0)
    for label,r in s['typed_twohop_closure'].items():
        for k,v in r.items():
            if not k.endswith('reason'):raw['closure_'+label+'_'+k]=v
    for i in range(3):
        raw['MP'+str(i+1)+'_total']=s['mp'][i]['total'];raw['MP'+str(i+1)+'_per_node']=s['mp'][i]['per_node'];raw['walk'+str(i+1)+'_total']=s['typed_walks'][i]['total']
    raw['powerlaw_fixed_xmin1_alpha']=s['powerlaw_fixed_xmin1_project']['alpha'];raw['powerlaw_fixed_xmin1_KS']=s['powerlaw_fixed_xmin1_project']['ks']
    distance={k:comp[k] for k in ['total_degree_W1','main_incident4_W1','all10_compat_W1','incident4_actual_typeweights','main_joint_degree_TV','triangle_composition_TV','four_cycle_composition_TV']}
    distance['degree_MMD2_alpha4_raw']=comp['degree_MMD2_alpha4']['raw'];distance['degree_MMD2_alpha4_nonnegative']=comp['degree_MMD2_alpha4']['nonnegative']
    for t,name in enumerate(NAMES):
        distance['type_'+name+'_total_degree_W1']=comp['type_total_degree_W1'][name];distance['type_'+name+'_joint_TV']=comp['joint_TV_by_type'][name]
        for r,v in enumerate(comp['typed_relation_W1'][name]):distance['type_'+name+'_relation'+str(r)+'_W1']=v
    for i in range(3):distance['MP'+str(i+1)+'_composition_TV']=comp['MP_composition_TV'][i];distance['walk'+str(i+1)+'_composition_TV']=comp['walk_composition_TV'][i]
    for r,v in comp['family_endpoint_W1'].items():distance['family'+r+'_endpoint_W1']=v
    for k in ['edge_overlap_count','EO_reference_denominator','EO_generated_denominator','EO_jaccard','edge_novelty_generated_denominator']:raw[k]=comp[k]
    return raw,distance

def main():
    started=time.time();p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--reference',required=True);p.add_argument('--output',required=True);a=p.parse_args();out=pathlib.Path(a.output);run=pathlib.Path(a.run);ledger=json.loads((out/'ledger.json').read_text())
    def check():
        if time.time()>=ledger['deadline_epoch']-180:raise TimeoutError('reserve final180 seconds for incomplete delivery')
    F=M.frozen(out/'vendor/frozen_metrics.py',out/'vendor/aligned_evaluate.py',out/'vendor/hgen_utils.py')
    with np.load(a.reference,allow_pickle=False) as z:ref=M.make_graph(z['types'],z['edges'].T,z['schema'])
    rs=M.stats(ref,F,NAMES,check);assert rs['nodes']==192 and rs['edges']==926 and rs['component_sizes']==[192],'reference connectivity differs; disclose and stop'
    save(out/'reference_metrics.json',rs);shutil.copy2(a.reference,out/'reference.npz');rc=M.compare(ref,rs,ref,rs,F,NAMES);ref_values,_=scalar(rs,rs,rc)
    records=[];retention=[];type_rows=[];relation_rows=[];tri_rows=[];path_rows=[];closure_rows=[];metric_rows=[];old_errors=[];inputs=[];graphs={};excluded_records=[]
    for seed,kind,sample in itertools.product(SEEDS,READOUTS,SAMPLES):
        check();identity=dict(train_seed=seed,cohort=cohort(seed),readout=kind,sample_seed=sample);folder=run/('PubMed_s%d_O'%seed)/kind/('sample_%d'%sample);path=folder/'graph.npz';binding=json.loads((folder/'binding.json').read_text());h=M.sha(path);assert h==binding['graph_sha256']
        with np.load(path,allow_pickle=False) as z:g=M.make_graph(z['types'],z['edges'],z['schema']);np.testing.assert_array_equal(g['families'],z['families'])
        target=out/'graphs'/('s%d'%seed)/kind/str(sample);target.mkdir(parents=True,exist_ok=False);shutil.copy2(path,target/'original.npz')
        kept,mapping=M.lcc(g);np.testing.assert_array_equal(kept['degree'],g['degree'][mapping['selected_local_ids']]);np.savez_compressed(target/'lcc.npz',types=kept['types'],edges=kept['edges'],families=kept['families'],schema=kept['schema'],new_to_original=kept['original_ids'],old_to_new=np.array(mapping['old_to_new']))
        mapping.update(source_sha256=h,derived_sha256=M.sha(target/'lcc.npz'),source_path=str(path),**identity);save(target/'mapping.json',mapping);inputs.append(dict(**identity,source=str(path),sha256=h,derived_sha256=mapping['derived_sha256']))
        stats={};scalars={};distances={}
        for version,graph in [('original',g),('lcc',kept)]:
            check();s=M.stats(graph,F,NAMES,check);comp=M.compare(graph,s,ref,rs,F,NAMES);raw,dist=scalar(s,rs,comp);stats[version]=s;scalars[version]=raw;distances[version]=dist;graphs[(seed,kind,sample,version)]=graph
            val=dict(**identity,version=version,statistics=s,comparison=comp,raw=raw,reference=ref_values,absolute_error={k:None if v is None or ref_values.get(k) is None else abs(v-ref_values[k]) for k,v in raw.items()},distribution_error=dist)
            val['undefined']={group:{k:'undefined_zero_opportunity_capacity_missing_type_or_unavailable_implementation' for k,v in val[group].items() if v is None} for group in ['raw','reference','absolute_error','distribution_error']}
            records.append(val);save(target/(version+'_metrics.json'),val)
            for group in ['raw','reference','absolute_error','distribution_error']:
                for k,v in val[group].items():metric_rows.append(dict(**identity,version=version,metric_group=group,metric=k,value=v,reason=val['undefined'][group].get(k)))
            for t,name in enumerate(NAMES):
                ix=graph['types']==t;ri=ref['types']==t;sp=s['typed_degree'][name];type_rows.append(dict(**identity,version=version,type=name,nodes=int(ix.sum()),reference_nodes=int(ri.sum()),share=float(ix.mean()),maximum=sp['maximum'],second=sp['second'],p90=sp['p90'],p99=sp['p99'],zero=sp['zero'],low_le1=sp['low_le1'],degree_gini=sp['gini_standard'],top5_degree_share=sp['top5_degree_share'],total_degree_W1=comp['type_total_degree_W1'][name],joint_TV=comp['joint_TV_by_type'][name],sorted_degrees=json.dumps(sp['sorted_degrees'])))
            for r,(ta,tb) in enumerate(graph['schema']):relation_rows.append(dict(**identity,version=version,family=r,label=NAMES[ta]+'--'+NAMES[tb],edges=s['family_edges'][r],reference_edges=rs['family_edges'][r],signed_error=s['family_edges'][r]-rs['family_edges'][r],absolute_error=abs(s['family_edges'][r]-rs['family_edges'][r]),capacity=s['capacities'][r],density=s['family_edges'][r]/s['capacities'][r] if s['capacities'][r] else None,reference_density=rs['family_edges'][r]/rs['capacities'][r]))
            for pattern in itertools.combinations_with_replacement(range(4),3):
                key=str(pattern);count=s['typed_triangles'].get(key,0);refcount=rs['typed_triangles'].get(key,0);tri_rows.append(dict(**identity,version=version,pattern=key,label='--'.join(NAMES[t] for t in pattern),count=count,reference=refcount,absolute_error=abs(count-refcount),share=count/s['triangles'] if s['triangles'] else None,reference_share=refcount/rs['triangles'],reference_zero=refcount==0))
            for mode,field in [('simple_path','mp'),('walk_allow_repeats','typed_walks')]:
                for i in range(3):
                    keys=set(s[field][i]['counts'])|set(rs[field][i]['counts'])
                    for key in sorted(keys):
                        count=s[field][i]['counts'].get(key,0);refcount=rs[field][i]['counts'].get(key,0);path_rows.append(dict(**identity,version=version,definition=mode,length=i+1,pattern=key,count=count,reference=refcount,absolute_error=abs(count-refcount),share=count/s[field][i]['total'] if s[field][i]['total'] else None,reference_share=refcount/rs[field][i]['total'] if rs[field][i]['total'] else None))
            for key,r in s['typed_twohop_closure'].items():closure_rows.append(dict(**identity,version=version,pattern=key,**r,reference=json.dumps(rs['typed_twohop_closure'][key])))
        old=json.loads((folder/'metrics.json').read_text());s=stats['original'];c=distances['original'];err=0.
        for key in ['nodes','edges','lcc','isolates','maximum_degree','triangles','average_clustering']:
            e=abs(s[key]-old[key]);err=max(err,e);assert e<1e-10,(identity,key,e)
        assert s['family_edges']==old['family_edges'] and s['typed_triangles']==old['typed_triangles'] and s['mp']==old['mp']
        for value,expected in [(c['main_incident4_W1'],old['W1_definitions']['incident4']),(c['main_joint_degree_TV'],old['primary']['type_weighted_joint_degree_TV']),(c['triangle_composition_TV'],old['primary']['typed_triangle_composition_TV'])]:assert abs(value-expected)<1e-10
        old_errors.append(err);s,k=stats['original'],stats['lcc'];row=dict(**identity,nodes_original=s['nodes'],nodes_lcc=k['nodes'],nodes_retained=k['nodes']/s['nodes'],edges_original=s['edges'],edges_lcc=k['edges'],edges_retained=k['edges']/s['edges'] if s['edges'] else None,components_original=s['components'],components_lcc=k['components'],isolates_original=s['isolates'],isolates_lcc=k['isolates'],second_component=s['second_component'],tie_count=mapping['tie_count'],triangles_lost=s['triangles']-k['triangles'],four_cycles_lost=s['four_cycles']-k['four_cycles'],MP2_lost=s['mp'][1]['total']-k['mp'][1]['total'],MP3_lost=s['mp'][2]['total']-k['mp'][2]['total'])
        for t,name in enumerate(NAMES):
            n,nk=int((g['types']==t).sum()),int((kept['types']==t).sum());row[name+'_lost']=n-nk;row[name+'_retained']=nk/n if n else None
        for r in range(10):row['family'+str(r)+'_edge_retention']=k['family_edges'][r]/s['family_edges'][r] if s['family_edges'][r] else None
        retention.append(row)
        for component in mapping['excluded']:
            ids=component['ids'];sub=g['graph'].subgraph(ids);losttypes=np.bincount(g['types'][ids],minlength=4);edgefamilies=Counter(g['adj'][u][v] for u,v in sub.edges());degree=g['total'][ids]
            item=dict(**identity,ids=ids,nodes=len(ids),edges=sub.number_of_edges(),type_counts=losttypes.tolist(),family_edges={str(k):v for k,v in edgefamilies.items()},maximum_degree=int(max(degree,default=0)),triangles=sum(nx for nx in __import__('networkx').triangles(sub).values())//3,reference_fixed_type_P99_hubs={name:int(sum(g['total'][u]>=max(1,math.ceil(rs['typed_degree'][name]['p99'])) for u in ids if g['types'][u]==t)) for t,name in enumerate(NAMES)})
            excluded_records.append(item)
        save(out/'progress.json',dict(completed_graphs=len(retention),expected_graphs=64,current=identity,elapsed_seconds=time.time()-started))
    # Fixed metric union, strict null propagation; no field intersections.
    index={(r['train_seed'],r['readout'],r['sample_seed'],r['version']):r for r in records};cell_rows=[];paired_rows=[];clean_rows=[];group_rows=[]
    groups=['raw','absolute_error','distribution_error']
    for seed,kind,version in itertools.product(SEEDS,READOUTS,['original','lcc']):
        rr=[index[seed,kind,k,version] for k in SAMPLES]
        for group in groups:
            keys=set().union(*(x[group] for x in records))
            for key in sorted(keys):
                values=[r[group].get(key) for r in rr];cell_rows.append(dict(train_seed=seed,cohort=cohort(seed),readout=kind,version=version,metric_group=group,metric=key,value=mean(values),reason=None if mean(values) is not None else 'one_or_more_required_values_undefined'))
    for seed,kind,sample in itertools.product(SEEDS,READOUTS,SAMPLES):
        a,b=index[seed,kind,sample,'lcc'],index[seed,kind,sample,'original']
        for group in groups:
            for key in sorted(set(a[group])|set(b[group])):clean_rows.append(dict(train_seed=seed,readout=kind,sample_seed=sample,contrast='lcc-original',metric_group=group,metric=key,difference=difference(a[group].get(key),b[group].get(key))))
    for seed,kind,version,group in itertools.product(SEEDS,READOUTS[1:],['original','lcc'],groups):
        for key in sorted(set(index[seed,kind,SAMPLES[0],version][group])|set(index[seed,'raw',SAMPLES[0],version][group])):
            ds=[difference(index[seed,kind,k,version][group].get(key),index[seed,'raw',k,version][group].get(key)) for k in SAMPLES];value=mean(ds)
            paired_rows.append(dict(train_seed=seed,cohort=cohort(seed),readout=kind,version=version,metric_group=group,metric=key,seed_difference=value,sample1_difference=ds[0],sample2_difference=ds[1],reason=None if value is not None else 'undefined_required_value'))
    for kind,version,group in itertools.product(READOUTS[1:],['original','lcc'],groups):
        selected=[r for r in paired_rows if r['readout']==kind and r['version']==version and r['metric_group']==group]
        for key in sorted({r['metric'] for r in selected}):
            for label,seeds in [('old_two',SEEDS[:2]),('new_six',SEEDS[2:]),('all_eight',SEEDS)]:
                values=[r['seed_difference'] for r in selected if r['metric']==key and r['train_seed'] in seeds];m=mean(values)
                group_rows.append(dict(readout=kind,version=version,cohort=label,metric_group=group,metric=key,mean_difference=m,median_difference=statistics.median(values) if m is not None else None,negative_count=sum(v<0 for v in values) if m is not None else None,zero_count=sum(v==0 for v in values) if m is not None else None,positive_count=sum(v>0 for v in values) if m is not None else None,expected_seeds=len(seeds),valid_seeds=sum(v is not None for v in values),reason=None if m is not None else 'one_or_more_required_seed_values_undefined'))
    diversity=[]
    for seed,kind,version in itertools.product(SEEDS,READOUTS,['original','lcc']):
        a,b=[graphs[seed,kind,k,version] for k in SAMPLES];ea=set(map(tuple,a['original_ids'][a['edges']].T));eb=set(map(tuple,b['original_ids'][b['edges']].T));na,nb=set(a['original_ids']),set(b['original_ids']);aa,bb=index[seed,kind,SAMPLES[0],version],index[seed,kind,SAMPLES[1],version]
        degree_w1=M.w1(a['total'],b['total']);tri_tv=F['tv'](Counter(aa['statistics']['typed_triangles']),Counter(bb['statistics']['typed_triangles']))
        diversity.append(dict(train_seed=seed,readout=kind,version=version,nodes1=len(na),nodes2=len(nb),node_jaccard=len(na&nb)/len(na|nb),labelled_edge_jaccard=len(ea&eb)/len(ea|eb) if ea|eb else None,labelled_edge_symmetric_difference=len(ea^eb),labelled_graphs_equal=na==nb and ea==eb,total_degree_W1_between_samples=degree_w1,triangle_composition_TV_between_samples=tri_tv,MP2_composition_TV_between_samples=F['tv'](Counter(aa['statistics']['mp'][1]['counts']),Counter(bb['statistics']['mp'][1]['counts'])),isomorphic_status='proven_nonisomorphic_by_size' if len(na)!=len(nb) or len(ea)!=len(eb) else 'not_determined_no_VF2_search',scope='original IDs align labelled comparisons; structural distribution differences do not prove population diversity'))
    outputs={'retention':retention,'types':type_rows,'relations':relation_rows,'triangles':tri_rows,'paths':path_rows,'closure':closure_rows,'metrics_long':metric_rows,'seed_means':cell_rows,'ema_pairs':paired_rows,'lcc_minus_original':clean_rows,'cohort_pairs':group_rows,'diversity':diversity}
    for name,rows in outputs.items():check();csv_save(out/(name+'.csv'),rows)
    save(out/'excluded_components.json',excluded_records);save(out/'input_manifest.json',inputs)
    save(out/'analysis.json',dict(status='complete',graph_count=64,versions=2,original_frozen_metrics_reproduced=True,max_original_scalar_error=max(old_errors),records=records,retention=retention,diversity=diversity,cohort_pairs=group_rows))
    result=dict(status='complete',seconds=time.time()-started,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,graphs=64,derived_lcc=64,metric_versions=128,original_frozen_checks=64,source_reference_connected=True,missing_types=sum(bool(r['comparison']['missing_types']) for r in records),ties=sum(r['tie_count']>1 for r in retention),training=0,forward=0,new_sampling=0,GPU=False)
    save(out/'execution.json',result);print(json.dumps(result))
if __name__=='__main__':main()
