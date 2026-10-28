"""Raw/reference/error summaries; missing metrics never disappear or get averaged away."""
import argparse,pathlib,json,sys,hashlib,math,time,itertools
import numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).parent/'package'))
SEEDS=(2026102821,2026102822)
DISTRIBUTIONS=('degree_W1_incident4','degree_W1_all10_compat','joint_degree_TV','typed_triangle_composition_TV','MP1_composition_TV','MP2_composition_TV','MP3_composition_TV')
CLOSURE=('path_numerator','path_denominator','path_ratio','unique_pair_numerator','unique_pair_denominator','unique_pair_ratio')
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def scalars(m):
    r={k:m.get(k) for k in ['nodes','edges','triangles','lcc','isolates','maximum_degree','average_clustering','assortativity']}
    r['components']=len(m['component_sizes']) if m.get('component_sizes') is not None else None
    for i,v in enumerate(m.get('family_edges',[])):r['family_edges_'+str(i)]=v
    for i in range(3):r['MP'+str(i+1)+'_total']=m['mp'][i].get('total') if len(m.get('mp',[]))>i and m['mp'][i] is not None else None
    for pattern in itertools.combinations_with_replacement(range(len(m.get('typed_degree',{}))),3):
        r['typed_triangle_'+str(pattern)]=(m.get('typed_triangles') or {}).get(str(pattern),0) if m.get('typed_triangles') is not None else None
    for name,t in m.get('typed_degree',{}).items():
        r['type_'+name+'_nodes']=t.get('count')
        for i,col in enumerate(t.get('by_relation',[])):
            for key in ['mean','max','zero','low_le1','p90','p99']:r['type_'+name+'_relation_'+str(i)+'_'+key]=col.get(key)
    if m.get('imdb_closure') is not None:r.update({'closure_'+k:m['imdb_closure'].get(k) for k in CLOSURE})
    if 'IMDB_focus' in m:r.update({k:m['IMDB_focus'].get(k) for k in ['AA','DAA','AAM']})
    return r
def finite(v):return isinstance(v,(int,float,np.integer,np.floating)) and not isinstance(v,(bool,np.bool_)) and math.isfinite(float(v))
def reason(m,k):
    if k.startswith('closure_') and k.endswith('_ratio'):
        denom='unique_pair_denominator' if 'unique_pair' in k else 'path_denominator'
        if (m.get('imdb_closure') or {}).get(denom)==0:return 'undefined_zero_closure_denominator'
    if k=='assortativity' and m.get(k) is None:return 'undefined_constant_degree_or_no_edges'
    return 'missing_or_nonfinite_metric'
def graph_summary(metrics,reference):
    raw=scalars(metrics);ref=scalars(reference);raw={k:raw.get(k) for k in ref}
    dist={k:None for k in DISTRIBUTIONS};w=metrics.get('W1_definitions',{});p=metrics.get('primary',{});comparison=metrics.get('comparison',{})
    dist.update(degree_W1_incident4=w.get('incident4'),degree_W1_all10_compat=w.get('all10_compat'),joint_degree_TV=p.get('type_weighted_joint_degree_TV'),typed_triangle_composition_TV=p.get('typed_triangle_composition_TV'))
    for i,value in enumerate(comparison.get('mp_frequency_TV',[])[:3]):dist['MP'+str(i+1)+'_composition_TV']=value
    groups=dict(raw=raw,reference=ref,absolute_error={},distribution_error=dist);undefined={g:{} for g in groups}
    for g,m in [('raw',metrics),('reference',reference)]:
        for k,v in groups[g].items():
            if not finite(v):groups[g][k]=None;undefined[g][k]=reason(m,k)
    for k in ref:
        if groups['raw'][k] is None or groups['reference'][k] is None:groups['absolute_error'][k]=None;undefined['absolute_error'][k]='undefined_or_missing_raw_or_reference'
        else:groups['absolute_error'][k]=abs(groups['raw'][k]-groups['reference'][k])
    for k,v in dist.items():
        if not finite(v):dist[k]=None;undefined['distribution_error'][k]='missing_or_nonfinite_distribution_error'
    return dict(**groups,undefined=undefined)
def strict_mean(rows):
    result={};undefined={}
    for group in sorted(set().union(*(r.keys() for r in rows))):
        if group=='undefined':continue
        result[group]={};undefined[group]={};keys=sorted(set().union(*(r.get(group,{}) for r in rows)))
        for k in keys:
            values=[r.get(group,{}).get(k) for r in rows]
            if not all(finite(v) for v in values):result[group][k]=None;undefined[group][k]='one_or_more_required_seed_metrics_undefined_or_missing'
            else:result[group][k]=sum(values)/len(values)
    result['undefined']=undefined;return result
def paired(left,right,template):
    fields={'raw_difference':'raw','absolute_error_difference':'absolute_error','distribution_error_difference':'distribution_error'};out={};undefined={}
    for target,source in fields.items():
        keys=set(template[source]) | set((left or {}).get(source,{})) | set((right or {}).get(source,{}));out[target]={};undefined[target]={}
        for k in sorted(keys):
            a=(left or {}).get(source,{}).get(k);b=(right or {}).get(source,{}).get(k)
            if not finite(a) or not finite(b):out[target][k]=None;undefined[target][k]='left_or_right_graph_or_metric_undefined_or_missing'
            else:out[target][k]=a-b
    return dict(**out,undefined=undefined)
