import numpy as np
import itertools,math
def scalar(s,ref,comp,names):
    raw={k:v for k,v in s.items() if (v is None or isinstance(v,(int,float,np.integer,np.floating))) and k not in ['HGEN_native_powerlaw']}
    raw['HGEN_native_powerlaw']=None
    for t,name in enumerate(names):
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
    for t,name in enumerate(names):
        distance['type_'+name+'_total_degree_W1']=comp['type_total_degree_W1'][name];distance['type_'+name+'_joint_TV']=comp['joint_TV_by_type'][name]
        for r,v in enumerate(comp['typed_relation_W1'][name]):distance['type_'+name+'_relation'+str(r)+'_W1']=v
    for i in range(3):distance['MP'+str(i+1)+'_composition_TV']=comp['MP_composition_TV'][i];distance['walk'+str(i+1)+'_composition_TV']=comp['walk_composition_TV'][i]
    for r,v in comp['family_endpoint_W1'].items():distance['family'+r+'_endpoint_W1']=v
    for k in ['edge_overlap_count','EO_reference_denominator','EO_generated_denominator','EO_jaccard','edge_novelty_generated_denominator']:raw[k]=comp[k]
    return raw,distance
