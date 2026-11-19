"""CPU-only frozen existing evaluation, separate terminal costs and score estimators."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,hashlib,time,csv,importlib.util,itertools
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'research_branches/seed_coverage_repair_release/experiments/seed_coverage_repair_v1'
sys.path.insert(0,str(OLD))
from lcc_eval import metrics as M
V=OLD/'lcc_eval/vendor';F=M.frozen(V/'frozen_metrics.py',V/'aligned_evaluate.py',V/'hgen_utils.py')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def csvsave(p,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
def flat(x,prefix=''):
    if isinstance(x,dict):return {k:v for a,b in x.items() for k,v in flat(b,prefix+'.'+str(a) if prefix else str(a)).items()}
    if isinstance(x,list):return {k:v for i,b in enumerate(x) for k,v in flat(b,prefix+'['+str(i)+']').items()}
    return {prefix:x}
def cosine(a,b):
    den=float(a.norm()*b.norm());return float(torch.dot(a,b)/den) if den else None
def main():
    out=Path(sys.argv[1]);protocol=json.loads((out/'protocol.json').read_text());torch.set_num_threads(1);start=time.time()
    spec=importlib.util.spec_from_file_location('score',out/'frozen_score.py');S=importlib.util.module_from_spec(spec);spec.loader.exec_module(S)
    def check():
        if time.time()>protocol['delivery_epoch']-60:raise TimeoutError('CPU evaluation delivery reserve')
    rows=[];scalars=[];cost_rows=[];pairs=[];estimators=[];group_rows=[];refs={};missing=[]
    for ds,m in protocol['models'].items():
        z=np.load(m['data']);types=z['types'];schema=z['schema'];edges=z['edges'];edges=edges.T if edges.shape[0]!=2 else edges
        names=['Director','Actor','Movie','Genre'] if ds=='IMDB' else ['Gene','Disease','Chemical','Species']
        ref=M.make_graph(types,edges,schema);rs=M.stats(ref,F,names,check);refs[ds]=rs
        a=np.zeros((len(types),len(types)),bool);a[edges[0],edges[1]]=True;a|=a.T;rf=S.features(a,types,schema);rx=S.descriptor(types,S.bfs(a))
        cg=[];costs=[];sg=[];valid_seeds=[]
        for ss in protocol['sample_seeds']:
            check();folder=out/ds/str(ss);gp=folder/'graph.npz'
            if not gp.exists():missing.append(dict(dataset=ds,sample_seed=ss,reason='no completed final graph'));continue
            z=np.load(gp);g=M.make_graph(z['types'],z['edges'],z['schema']);assert np.array_equal(g['types'],types) and np.array_equal(g['schema'],schema)
            metrics=folder/'full_metrics.json'
            if metrics.exists():full=json.loads(metrics.read_text())
            else:
                st=M.stats(g,F,names,check);cmp=M.compare(g,st,ref,rs,F,names);lg,mapping=M.lcc(g);ls=M.stats(lg,F,names,check);lc=M.compare(lg,ls,ref,rs,F,names)
                retention=dict(nodes=ls['nodes']/st['nodes'],edges=ls['edges']/st['edges'] if st['edges'] else None,by_type={name:int(np.sum(lg['types']==i))/int(np.sum(types==i)) for i,name in enumerate(names)},mapping=mapping)
                full=dict(raw=dict(statistics=st,comparison=cmp),LCC=dict(statistics=ls,comparison=lc,retention=retention),reference=rs,LCC_note='Derived postprocessing; guaranteed connectivity is not model ability; comparison against original reference.')
                save(metrics,full)
            for view in ['raw','LCC']:
                st=full[view]['statistics'];cmp=full[view]['comparison']
                rows.append(dict(dataset=ds,train_seed=m['train_seed'],sample_seed=ss,view=view,nodes=st['nodes'],edges=st['edges'],reference_edges=rs['edges'],edge_abs_error=abs(st['edges']-rs['edges']),incident4_W1=cmp['main_incident4_W1'],joint_TV=cmp['main_joint_degree_TV'],triangles=st['triangles'],reference_triangles=rs['triangles'],triangle_abs_error=abs(st['triangles']-rs['triangles']),triangle_composition_TV=cmp['triangle_composition_TV'],MP2=st['mp'][1]['total'],MP2_abs_error=abs(st['mp'][1]['total']-rs['mp'][1]['total']),MP2_TV=cmp['MP_composition_TV'][1],MP3=st['mp'][2]['total'],MP3_abs_error=abs(st['mp'][2]['total']-rs['mp'][2]['total']),MP3_TV=cmp['MP_composition_TV'][2],lcc=st['lcc'],components=st['components'],isolates=st['isolates'],four_cycles=st['four_cycles']))
                sc=flat(full[view]);scalars.extend(dict(dataset=ds,train_seed=m['train_seed'],sample_seed=ss,view=view,metric=k,value=v) for k,v in sc.items())
            aa=np.zeros_like(a);aa[g['edges'][0],g['edges'][1]]=True;aa|=aa.T
            terms=S.terms(S.features(aa,types,schema),rf,types,schema);terms['component_distance']=S.dist(S.descriptor(types,S.bfs(aa)),rx,types)
            cost_rows.append(dict(dataset=ds,sample_seed=ss,**terms));costs.append([terms[k] for k in protocol['score_terms']]);cg.append(g);valid_seeds.append(ss)
            gpfile=folder/'score_gradients.pt'
            if gpfile.exists():
                q=torch.load(gpfile,map_location='cpu');assert q['gradients'].shape[0]==8
                sg.append(q['gradients'].double().sum(0)*(1000/8));del q
        for i,j in itertools.combinations(range(len(cg)),2):
            x=set(map(tuple,cg[i]['edges'].T));y=set(map(tuple,cg[j]['edges'].T));pairs.append(dict(dataset=ds,seed1=valid_seeds[i],seed2=valid_seeds[j],intersection=len(x&y),union=len(x|y),jaccard=len(x&y)/len(x|y) if x|y else None,identical=x==y))
        if len(sg)==4 and len(costs)==4:
            G=torch.stack(sg);C=torch.tensor(costs,dtype=torch.float64);term_vectors=[]
            parameters=json.loads((out/ds/'parameters.json').read_text());groups={};offset=0
            for param in parameters:
                groups.setdefault(param['group'],[]).append((offset,offset+param['numel']));offset+=param['numel']
            assert offset==G.shape[1]
            for k,term in enumerate(protocol['score_terms']):
                c=C[:,k];base=(c.sum()-c)/3;contrib=(c-base)[:,None]*G;estimate=contrib.mean(0);term_vectors.append(estimate)
                half1=(c[0]-c[1])*(G[0]-G[1])/2;half2=(c[2]-c[3])*(G[2]-G[3])/2
                uncentered=(c[:,None]*G).mean(0)
                norm=float(estimate.norm());spread=float(((contrib-estimate).square().sum(1).mean()).sqrt())
                estimators.append(dict(dataset=ds,term=term,cost_mean=float(c.mean()),cost_sample_sd=float(c.std()),gradient_norm=norm,contribution_rms_spread=spread,spread_over_mean_norm=spread/norm if norm else None,two_disjoint_pairs_cosine=cosine(half1,half2),pair1_cost_difference=float(c[0]-c[1]),pair2_cost_difference=float(c[2]-c[3]),pair_score_geometry_cosine=cosine(G[0]-G[1],G[2]-G[3]),LOO_vs_uncentered_cosine=cosine(estimate,uncentered),note='Descriptive four-path diagnostics; LOO contributions share baseline, not independent replicates or confidence intervals. Two-path LOO is (C1-C2)*(S1-S2)/2, so all terms share the pair geometry up to signs; seven terms are not seven independent confirmations. Time subsampling noise included.'))
                for group,slices in groups.items():
                    select=lambda v:torch.cat([v[a:b] for a,b in slices])
                    ge=select(estimate);gh1=select(half1);gh2=select(half2)
                    group_rows.append(dict(dataset=ds,term=term,parameter_group=group,parameter_count=len(ge),gradient_norm=float(ge.norm()),norm_fraction=float(ge.norm())/norm if norm else None,two_disjoint_pairs_cosine=cosine(gh1,gh2)))
            mat=[[cosine(x,y) for y in term_vectors] for x in term_vectors];save(out/(ds+'_gradient_cosines.json'),dict(terms=protocol['score_terms'],cosines=mat))
            # Preserve estimates for audit, not as authorized optimizer inputs.
            torch.save(dict(terms=protocol['score_terms'],terminal_costs=C,path_score_gradients=G,estimated_gradients=torch.stack(term_vectors)),out/(ds+'_gradient_estimates.pt'))
            del G,sg,term_vectors
    summaries=[]
    for ds in protocol['models']:
        for view in ['raw','LCC']:
            subset=[r for r in rows if r['dataset']==ds and r['view']==view]
            if not subset:continue
            summary=dict(dataset=ds,view=view,training_models=1,samples=len(subset))
            for k in subset[0]:
                if k in ['dataset','view','train_seed','sample_seed']:continue
                values=[r[k] for r in subset if isinstance(r[k],(int,float))]
                summary[k+'_mean']=float(np.mean(values)) if values else None;summary[k+'_sample_sd']=float(np.std(values,ddof=1)) if len(values)>1 else None
            summaries.append(summary)
    for name,table in [('per_graph',rows),('all_metrics',scalars),('terminal_costs',cost_rows),('sample_pairs',pairs),('gradient_diagnostics',estimators),('gradient_groups',group_rows),('model_summary',summaries)]:save(out/(name+'.json'),table);csvsave(out/(name+'.csv'),table)
    save(out/'reference_metrics.json',refs);save(out/'evaluation_receipt.json',dict(elapsed_seconds=time.time()-start,completed_graphs=len(cost_rows),missing=missing,evaluator_hashes={str(p):sha(p) for p in [Path(__file__),OLD/'lcc_eval/metrics.py',*V.glob('*.py')]},GPU=0))
    print(json.dumps(dict(completed_graphs=len(cost_rows),missing=missing,seconds=time.time()-start)))
if __name__=='__main__':main()
