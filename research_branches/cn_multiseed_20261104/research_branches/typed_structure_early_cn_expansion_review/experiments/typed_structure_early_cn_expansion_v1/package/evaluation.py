"""Frozen raw-graph evaluation, no graph repair or model selection."""
from collections import Counter
import numpy as np
from audit_reference.role_condition.metrics import graph,compare,tv

def evaluate(case,edges,type_names,check=lambda:None):
    result=graph(case,edges,type_names,check);reference=graph(case,case['clean'],type_names,check);comparison=compare(result,reference)
    types=case['space'].types;w1=all_columns=joint=0.;per_type={}
    for typ,name in enumerate(type_names):
        rel=[r for r,pair in enumerate(case['space'].pairs) if typ in pair]
        weight=float(np.mean(types==typ));value=float(np.mean(np.array(comparison['typed_relation_degree_W1'][name])[rel]))
        w1+=weight*value;joint+=weight*comparison['joint_relation_degree_TV'][name]
        values=np.asarray(comparison['typed_relation_degree_W1'][name],dtype=float)
        nonincident=[r for r in range(case['n_families']) if r not in rel]
        if np.any(values[nonincident]!=0):raise ValueError('nonincident relation W1 must be zero')
        all_columns+=weight*float(np.mean(values))
        per_type[name]=dict(weight=weight,incident_relations=rel,mean_incident_relation_W1=value)
    tri_tv=tv(Counter(result['typed_triangles']),Counter(reference['typed_triangles']))
    result.update(comparison=comparison,primary=dict(type_weighted_relation_degree_W1=w1,type_weighted_joint_degree_TV=joint,typed_triangle_composition_TV=tri_tv),weight_definition=per_type)
    ratio=.4 if len(type_names)==4 and case['n_families']==10 and all(len(x['incident_relations'])==4 for x in per_type.values()) else None
    if ratio is not None and not np.isclose(all_columns,ratio*w1,atol=1e-12,rtol=1e-12):raise ValueError('incident4/all10 W1 consistency failure')
    result['W1_definitions']=dict(main='incident4',incident4=w1,all10_compat=all_columns,all10_to_incident4_ratio=ratio,scope='all10 names apply to the two frozen 4-type/10-family datasets; toy schemas use actual column count')
    if type_names==['Director','Actor','Movie','Genre']:
        aa=case['space'].pairs.index((1,1));result['IMDB_focus']=dict(AA=result['family_edges'][aa],DAA=result['typed_triangles'].get(str((0,1,1)),0),AAM=result['typed_triangles'].get(str((1,1,2)),0))
        reference['IMDB_focus']=dict(AA=reference['family_edges'][aa],DAA=reference['typed_triangles'].get(str((0,1,1)),0),AAM=reference['typed_triangles'].get(str((1,1,2)),0))
    result['reference_metrics']=reference
    return result

def paired_table(rows):
    """Each row key=(dataset,arm,sample_seed); retain all rows and signs."""
    diffs=[]
    for dataset in ['IMDB','PubMed']:
        for seed in [2026102821,2026102822]:
            for left,right in [('A','O'),('B','O'),('B','A'),('C','B'),('C','A')]:
                a=rows[(dataset,left,seed)]['primary'];b=rows[(dataset,right,seed)]['primary']
                diffs.append(dict(dataset=dataset,seed=seed,contrast=left+'-'+right,primary={k:a[k]-b[k] for k in a}))
    return diffs
