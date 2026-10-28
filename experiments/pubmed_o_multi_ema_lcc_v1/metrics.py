"""CPU statistics on variable-size typed graphs; no model or torch import."""
import ast,collections,hashlib,itertools,json,math,pathlib,types
import numpy as np
import networkx as nx
from scipy.sparse.csgraph import shortest_path
from scipy.special import zeta
Counter=collections.Counter

def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def load_functions(path,names,namespace):
    tree=ast.parse(pathlib.Path(path).read_text());nodes=[x for x in tree.body if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef)) and x.name in names]
    assert {x.name for x in nodes}==set(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    return {k:namespace[k] for k in names}
def frozen(frozen_metrics,aligned,hgen):
    a=load_functions(frozen_metrics,['canon','paths','tv'],dict(Counter=Counter,np=np))
    b=load_functions(aligned,['walks','mmd','powerlaw'],dict(collections=collections,Counter=Counter,np=np,zeta=zeta,canon=a['canon']))
    h=load_functions(hgen,['statistics_gini','statistics_edge_distribution_entropy'],dict(np=np))
    return dict(**a,**b,**h)

def make_graph(types,edges,schema,original_ids=None):
    types=np.asarray(types,np.int64);edges=np.asarray(edges,np.int64).reshape(2,-1);schema=np.asarray(schema,np.int64)
    assert len(types)>0 and np.all(edges>=0) and np.all(edges<len(types)) and np.all(edges[0]<edges[1])
    assert len(set(map(tuple,edges.T)))==edges.shape[1]
    lookup={tuple(p):i for i,p in enumerate(schema)}
    fam=np.array([lookup[tuple(sorted((int(types[u]),int(types[v]))))] for u,v in edges.T],np.int64)
    degree=np.zeros((len(types),len(schema)),np.int64);np.add.at(degree,(edges[0],fam),1);np.add.at(degree,(edges[1],fam),1)
    adj=[{} for _ in types]
    for (u,v),r in zip(edges.T,fam):adj[u][int(v)]=int(r);adj[v][int(u)]=int(r)
    graph=nx.Graph();graph.add_nodes_from(range(len(types)));graph.add_edges_from(edges.T)
    components=sorted((sorted(c) for c in nx.connected_components(graph)),key=lambda c:(-len(c),c))
    return dict(types=types,edges=edges,schema=schema,families=fam,degree=degree,total=degree.sum(1),adj=adj,graph=graph,components=components,original_ids=np.arange(len(types)) if original_ids is None else np.asarray(original_ids,np.int64))

def lcc(g):
    # Storage-ID lexicographic tie-break; never reference similarity or node features.
    components=sorted(g['components'],key=lambda c:(-len(c),sorted(map(int,g['original_ids'][c]))))
    keep=np.array(sorted(components[0]),np.int64);old_to_new=np.full(len(g['types']),-1,np.int64);old_to_new[keep]=np.arange(len(keep))
    mask=np.isin(g['edges'][0],keep)&np.isin(g['edges'][1],keep);edges=old_to_new[g['edges'][:,mask]]
    out=make_graph(g['types'][keep],edges,g['schema'],g['original_ids'][keep])
    ties=[c for c in components if len(c)==len(keep)]
    mapping=dict(new_to_original=g['original_ids'][keep].tolist(),old_to_new=old_to_new.tolist(),selected_local_ids=keep.tolist(),tie_count=len(ties),tied_components=[g['original_ids'][c].tolist() for c in ties] if len(ties)>1 else [],rule='largest node count; lexicographically smallest sorted original storage IDs',excluded=[dict(ids=g['original_ids'][c].tolist(),size=len(c)) for c in components[1:]])
    assert set(map(tuple,out['original_ids'][out['edges']].T))<=set(map(tuple,g['original_ids'][g['edges']].T))
    return out,mapping

def capacity(g):
    counts=np.bincount(g['types'],minlength=4)
    return [int(counts[a]*(counts[a]-1)//2 if a==b else counts[a]*counts[b]) for a,b in g['schema']]
def hist(v):return np.bincount(np.asarray(v,dtype=int)).tolist() if len(v) else []
def w1(a,b):
    if not len(a) or not len(b):return None
    n=int(max(max(a),max(b)))+1
    return float(np.abs(np.cumsum(np.bincount(a,minlength=n)/len(a)-np.bincount(b,minlength=n)/len(b))).sum())
def gini(v):
    v=np.sort(np.asarray(v,float));n=len(v)
    return float(2*np.sum(np.arange(1,n+1)*v)/(n*v.sum())-(n+1)/n) if n and v.sum() else None
def spectrum(v):
    v=np.asarray(v,int);n=len(v);total=int(v.sum());order=sorted(map(int,v),reverse=True)
    return dict(count=n,histogram=hist(v),sorted_degrees=order,mean=float(v.mean()) if n else None,maximum=order[0] if n else None,second=order[1] if n>1 else None,zero=int((v==0).sum()),low_le1=int((v<=1).sum()),p90=float(np.quantile(v,.9)) if n else None,p99=float(np.quantile(v,.99)) if n else None,gini_standard=gini(v),top5_degree_share=sum(order[:5])/total if total else None)
def cycle_signature(labels):
    # labels=(node type, outgoing relation) around the cycle.
    out=[];m=len(labels);ts=[x[0] for x in labels];rs=[x[1] for x in labels]
    for direction in [1,-1]:
        for start in range(m):
            order=[(start+direction*k)%m for k in range(m)]
            out.append(tuple((ts[i],rs[i] if direction==1 else rs[(i-1)%m]) for i in order))
    return repr(min(out))

def extras(g,check):
    types,adj,edges=g['types'],g['adj'],g['edges'];n=len(types);A=nx.to_numpy_array(g['graph'],nodelist=range(n),dtype=float);d=g['total'];common=A.astype(np.int64)@A.astype(np.int64)
    closure={};four=Counter();k4=0
    for middle in range(4):
        for a,c in itertools.combinations_with_replacement(range(4),2):closure[str((a,middle,c))]=dict(path_numerator=0,path_denominator=0,unique_pair_numerator=0,unique_pair_denominator=0)
    for u in range(n):
        if u%20==0:check()
        for v in range(u+1,n):
            neighbors=adj[u].keys()&adj[v].keys()
            if not neighbors:continue
            a,c=sorted((int(types[u]),int(types[v])));counts=Counter(int(types[w]) for w in neighbors)
            for middle,count in counts.items():
                r=closure[str((a,middle,c))];r['path_denominator']+=count;r['unique_pair_denominator']+=1
                if v in adj[u]:r['path_numerator']+=count;r['unique_pair_numerator']+=1
            categories=Counter((int(types[w]),adj[u][w],adj[v][w]) for w in neighbors);items=list(categories.items())
            for i,((tw,uw,vw),ca) in enumerate(items):
                for j in range(i,len(items)):
                    (tx,ux,vx),cb=items[j];count=ca*cb if i!=j else ca*(ca-1)//2
                    if count:four[cycle_signature([(int(types[u]),uw),(tw,vw),(int(types[v]),vx),(tx,ux)])]+=count
            if v in adj[u]:
                nodes=sorted(w for w in neighbors if w>v)
                for w in nodes:k4+=sum(x>w and x in adj[w] for x in nodes)
    assert all(c%2==0 for c in four.values());four={k:int(v//2) for k,v in sorted(four.items())}
    for r in closure.values():
        for label in ['path','unique_pair']:
            den=r[label+'_denominator'];r[label+'_ratio']=r[label+'_numerator']/den if den else None
            r[label+'_reason']=None if den else 'undefined_zero_opportunities'
    distances=shortest_path(A,directed=False,unweighted=True);mask=~np.eye(n,dtype=bool)&np.isfinite(distances);values=distances[mask]
    ids=g['components'][0];sub=distances[np.ix_(ids,ids)];sv=sub[~np.eye(len(ids),dtype=bool)]
    return dict(typed_twohop_closure=closure,typed_four_cycles=four,four_cycles=sum(four.values()),HGEN_square_legacy_K4=k4,claw_count=int(sum(int(x)*(int(x)-1)*(int(x)-2)//6 for x in d)),wedge_count=int(sum(int(x)*(int(x)-1)//2 for x in d)),gini_standard=gini(d),cpl_finite_ordered_pairs=float(values.mean()) if len(values) else None,cpl_pair_count=int(len(values)),reachable_pair_fraction=len(values)/(n*(n-1)) if n>1 else None,cpl_lcc=float(sv.mean()) if len(sv) else None,diameter_global=int(values.max()) if len(g['components'])==1 and len(values) else None,diameter_max_component_finite=int(values.max()) if len(values) else None,diameter_lcc=int(sv.max()) if len(sv) else None,distance_histogram=hist(values.astype(int)),A=A)

def stats(g,F,names,check):
    types,edges,d=g['types'],g['edges'],g['degree'];mp,tr,adj=F['paths'](types,edges,g['families'],check)
    walks=F['walks'](g);more=extras(g,check);A=more.pop('A');s=dict(nodes=len(types),edges=edges.shape[1],family_edges=np.bincount(g['families'],minlength=10).tolist(),capacities=capacity(g),components=len(g['components']),component_sizes=[len(c) for c in g['components']],lcc=len(g['components'][0]),second_component=len(g['components'][1]) if len(g['components'])>1 else 0,isolates=int((g['total']==0).sum()),maximum_degree=int(g['total'].max()),minimum_degree=int(g['total'].min()),mean_degree=float(g['total'].mean()),total_degree=spectrum(g['total']),total_degree_histogram=hist(g['total']),typed_degree={},joint_relation_degree={},triangles=int(sum(tr.values())),typed_triangles={str(k):int(v) for k,v in sorted(tr.items())},mp=[dict(counts={','.join(map(str,k)):int(v) for k,v in sorted(c.items())},total=int(sum(c.values())),per_node=sum(c.values())/len(types)) for c in mp],typed_walks=[dict(counts={','.join(map(str,k)):int(v) for k,v in sorted(c.items())},total=int(sum(c.values()))) for c in walks],average_clustering=float(nx.average_clustering(g['graph'])),**more)
    s['transitivity']=3*s['triangles']/s['wedge_count'] if s['wedge_count'] else None
    s['HGEN_clustering_legacy_3T_over_claws']=3*s['triangles']/s['claw_count'] if s['claw_count'] else None
    s['HGEN_gini_legacy_zero_based']=float(F['statistics_gini'](A)) if g['total'].sum() else None
    s['HGEN_entropy_legacy_eps1e_4']=float(F['statistics_edge_distribution_entropy'](A)) if len(types)>1 and g['total'].sum() else None
    s['powerlaw_fixed_xmin1_project']=F['powerlaw'](g['total']);s['HGEN_native_powerlaw']=None;s['HGEN_native_powerlaw_reason']='powerlaw package unavailable; no install or substitute'
    if edges.shape[1]:
        x=g['total'][edges].astype(float);mean=float(x.mean());var=float((x*x).mean()-mean*mean);s['assortativity']=float((np.mean(x[0]*x[1])-mean*mean)/var) if var>0 else None
    else:s['assortativity']=None
    for t,name in enumerate(names):
        rows=d[types==t];sp=spectrum(g['total'][types==t]);sp.update(by_relation=[])
        for col in rows.T:
            r=spectrum(col);sp['by_relation'].append(dict(histogram=r['histogram'],mean=r['mean'],max=r['maximum'],zero=r['zero'],low_le1=r['low_le1'],p90=r['p90'],p99=r['p99']))
        s['typed_degree'][name]=sp;s['joint_relation_degree'][name]={','.join(map(str,k)):int(v) for k,v in sorted(Counter(map(tuple,rows)).items())}
    # Original normalization and all 40 typed opportunities are kept separately.
    assert sum(r['path_denominator'] for r in s['typed_twohop_closure'].values())==s['wedge_count']
    assert sum(r['path_numerator'] for r in s['typed_twohop_closure'].values())==3*s['triangles']
    return s

def compare(g,s,ref,rs,F,names):
    result=dict(total_degree_W1=w1(g['total'],ref['total']),degree_MMD2_alpha4=F['mmd'](g['total'],ref['total']),typed_relation_W1={},joint_TV_by_type={},type_total_degree_W1={},type_total_degree_MMD2={},family_endpoint_W1={},missing_types=[],main_incident4_W1=None,all10_compat_W1=None,incident4_actual_typeweights=None,main_joint_degree_TV=None,triangle_composition_TV=F['tv'](Counter(s['typed_triangles']),Counter(rs['typed_triangles'])),four_cycle_composition_TV=F['tv'](Counter(s['typed_four_cycles']),Counter(rs['typed_four_cycles'])),MP_composition_TV=[F['tv'](Counter(a['counts']),Counter(b['counts'])) for a,b in zip(s['mp'],rs['mp'])],walk_composition_TV=[F['tv'](Counter(a['counts']),Counter(b['counts'])) for a,b in zip(s['typed_walks'],rs['typed_walks'])])
    fixed=current=all10=joint=0.
    for t,name in enumerate(names):
        ix,iy=g['types']==t,ref['types']==t;rows=g['degree'][ix];rrows=ref['degree'][iy]
        if not len(rows):result['missing_types'].append(name);result['typed_relation_W1'][name]=[None]*10;result['joint_TV_by_type'][name]=None;result['type_total_degree_W1'][name]=None;result['type_total_degree_MMD2'][name]=dict(raw=None,nonnegative=None);continue
        values=[w1(a,b) for a,b in zip(rows.T,rrows.T)];rel=[r for r,p in enumerate(g['schema']) if t in p]
        value=float(np.mean(np.array(values)[rel]));fw=float(iy.mean());cw=float(ix.mean());j=F['tv'](Counter(s['joint_relation_degree'][name]),Counter(rs['joint_relation_degree'][name]));fixed+=fw*value;current+=cw*value;all10+=fw*np.mean(values);joint+=fw*j
        result['typed_relation_W1'][name]=values;result['joint_TV_by_type'][name]=j;result['type_total_degree_W1'][name]=w1(g['total'][ix],ref['total'][iy]);result['type_total_degree_MMD2'][name]=F['mmd'](g['total'][ix],ref['total'][iy])
    if not result['missing_types']:
        result.update(main_incident4_W1=float(fixed),all10_compat_W1=float(all10),incident4_actual_typeweights=float(current),main_joint_degree_TV=float(joint));assert math.isclose(all10,.4*fixed,abs_tol=1e-12,rel_tol=1e-12)
    result['distribution_reason']='undefined_missing_type' if result['missing_types'] else None
    for r,(a,b) in enumerate(g['schema']):
        x=(g['types']==a)|(g['types']==b);y=(ref['types']==a)|(ref['types']==b);result['family_endpoint_W1'][str(r)]=w1(g['degree'][x,r],ref['degree'][y,r])
    orig=set(map(tuple,g['original_ids'][g['edges']].T));truth=set(map(tuple,ref['original_ids'][ref['edges']].T));overlap=len(orig&truth)
    result.update(edge_overlap_count=overlap,EO_reference_denominator=overlap/len(truth) if truth else None,EO_generated_denominator=overlap/len(orig) if orig else None,EO_jaccard=overlap/len(orig|truth) if orig|truth else None,edge_novelty_generated_denominator=len(orig-truth)/len(orig) if orig else None)
    return result
