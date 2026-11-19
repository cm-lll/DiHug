"""CPU-only readout of previously computed edge counterfactuals. No model calls."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import csv,json,time,hashlib,resource
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)
R=Path(__file__).resolve().parents[2]
O=R/'results/structure_tradeoff_cpu_20261108'
E=R/'results/hard_structure_edge_direction_cpu_20261108'
J=R/'results/joint_transport_cpu_20261108'
P=R/'results/hard_structure_response_20261108T182154'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
save=lambda n,x:(O/n).write_text(json.dumps(x,indent=2))
def csvsave(name,rows):
    with (O/name).open('w') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
def classify(d):
    neg=d < -1e-12; pos=d > 1e-12
    return dict(nonworse_improvement=(~pos.any(1))&neg.any(1),
                strict_all_improvement=neg.all(1),
                nonbetter_damage=(~neg.any(1))&pos.any(1),
                strict_all_damage=pos.all(1),
                tradeoff=pos.any(1)&neg.any(1),neutral=~(pos|neg).any(1),
                triangle_improves_others_nonworse=neg[:,0]&~pos[:,1:].any(1))
def counts(d,s,weight=None):
    if weight is None:weight=np.ones(len(s))
    masks=classify(d); good=s < -1e-12
    out={'slots':len(s),'score_improves':float(weight[good].sum())}
    for k,m in masks.items():
        out[k]=float(weight[m].sum())
        out['rewarded_'+k]=float(weight[m&good].sum())
    out['missed_nonworse_improvement']=float(weight[masks['nonworse_improvement']&~good].sum())
    assert np.all(sum(masks[k].astype(int) for k in ['nonworse_improvement','nonbetter_damage','tradeoff','neutral'])==1)
    return out

def main():
    start=time.time();protocol=read(O/'protocol.json');bound={}
    def bind(p):bound[str(p)]=sha(p);return p
    rows=list(csv.DictReader(bind(E/'graph_summary.csv').open()));tasks=read(bind(R/'results/output_boundary_replay_20261108T081654/tasks.json'))
    datasets={};output=[];decomp=[]
    for ds in ['IMDB','PubMed']:
        task=next(x for x in tasks if x['dataset']==ds);cfg=task['row']['config']['datasets'][ds]
        path=R/'research_branches/typed_structure_early_cn_expansion_review/experiments/typed_structure_early_cn_expansion_v1/package'/cfg['file']
        assert sha(path)==cfg['sha256'];bind(path)
        with np.load(path) as z:types=z['types'].copy()
        inputs=torch.load(bind(P/f'{ds}_fixed_inputs.pt'),map_location='cpu')['inputs']
        datasets[ds]={}
        reference_sets=[]
        for t,_,inp in inputs:
            q=np.asarray(inp['query']);y=np.asarray(inp['labels']);families=np.asarray(inp['f'])
            datasets[ds][int(t)]=(types,q,y,families)
            reference_sets.append(set(map(tuple,q[:,y==1].T.tolist())))
        assert all(s==reference_sets[0] for s in reference_sets)
    for row in rows:
        assert time.time()<protocol['science_stop_epoch']
        ds=row['condition'].split('_')[0];types,q,y,f=datasets[ds][int(row['t'])]
        filename=f"{row['condition']}_{row['arm']}_t{row['t']}_draw{row['draw']}_flips.npz"
        with np.load(bind(E/filename)) as z:
            d=np.stack([z[k].copy() for k in ['triangle_abs_error_flip','total_degree_W1_flip','joint_degree_TV_flip']],axis=1)
            score=z['score_flip'].copy();sign=z['flip_sign'].copy()
        ids={k:row[k] for k in ['condition','arm','t','draw','triangle_state']};ids['dataset']=ds
        output.append(dict(**ids,method='MMD',panel='all_slots',**counts(d,score)))
        if row['condition']==ds+'_CE' and row['arm']=='CE' and row['t']=='500' and row['draw']=='0':
            with np.load(bind(J/f'{ds}_full_cost_map.npz')) as z:s=z['JOINT_W1_flip'].copy()
            output.append(dict(**ids,method='JOINT_W1',panel='two_full_benchmarks',**counts(d,s)))
        # Exact lower-bound decomposition: degree W1 = absolute type mean difference + residual.
        b=(1-sign)/2;nt=len(np.unique(types));mean_base=0.;mean_delta=np.zeros(len(b));check_w=0.
        degree=np.zeros(len(types));reference_degree=np.zeros(len(types))
        for endpoint in q:
            np.add.at(degree,endpoint,b);np.add.at(reference_degree,endpoint,y)
        for t in np.unique(types):
            n=int((types==t).sum());multiplicity=(types[q[0]]==t).astype(int)+(types[q[1]]==t).astype(int)
            diff=float(np.dot(multiplicity,b-y))/n
            check_w+=np.mean(abs(np.sort(degree[types==t])-np.sort(reference_degree[types==t])))/nt
            mean_base+=abs(diff)/nt
            mean_delta+=(abs(diff+sign*multiplicity/n)-abs(diff))/nt
        base_w=float(row['degree_W1']);residual=base_w-mean_base
        assert abs(check_w-base_w)<1e-10,(ids,check_w,base_w)
        assert residual>=-1e-10
        assert np.all(base_w+d[:,1]-(mean_base+mean_delta)>=-1e-10)
        residual_delta=d[:,1]-mean_delta;reward=score < -1e-12;triup=d[:,0]>1e-12
        decomp.append(dict(**ids,degree_W1=base_w,type_mean_gap=mean_base,residual=residual,
            rewarded_triangle_worse=int((reward&triup).sum()),
            thereof_degree_better=int((reward&triup&(d[:,1]<-1e-12)).sum()),
            thereof_degree_and_joint_nonbetter=int((reward&triup&(d[:,1]>=-1e-12)&(d[:,2]>=-1e-12)).sum()),
            rewarded_mean_worse_residual_better=int((reward&(mean_delta>1e-12)&(residual_delta<-1e-12)).sum()),
            rewarded_mean_better_residual_worse=int((reward&(mean_delta<-1e-12)&(residual_delta>1e-12)).sum())))
    csvsave('graph_directions.csv',output);csvsave('degree_decomposition.csv',decomp)
    # Same stratified edge panel, same inverse inclusion weights as the completed ticket.
    selected=list(csv.DictReader(bind(J/'selected_edges.csv').open()));strat=[]
    from collections import defaultdict
    groups=defaultdict(list)
    for x in selected:groups[(x['condition'],x['arm'],x['t'],x['draw'],x['triangle_state'])].append(x)
    for key,g in groups.items():
        d=np.array([[float(x[k]) for k in ['triangle_abs_error_flip','total_degree_W1_flip','joint_degree_TV_flip']] for x in g]);w=np.array([float(x['inverse_inclusion_probability']) for x in g])
        for method in ['MMD','JOINT_W1']:
            s=np.array([float(x[method+'_flip']) for x in g]);ids=dict(zip(['condition','arm','t','draw','triangle_state'],key));ids['dataset']=key[0].split('_')[0]
            strat.append(dict(**ids,method=method,panel='stratified_HT',**counts(d,s,w)))
    csvsave('stratified_directions.csv',strat)
    summary=[]
    for panel in [output,strat]:
        groups=defaultdict(list)
        for x in panel:groups[(x['dataset'],x['method'],x['panel'],x['t'])].append(x)
        for key,g in groups.items():
            ids=dict(zip(['dataset','method','panel','t'],key));tot={k:sum(x[k] for x in g) for k in g[0] if isinstance(g[0][k],(float,int))}
            summary.append(dict(**ids,graphs=len(g),graphs_with_nonworse_improvement=sum(x['nonworse_improvement']>0 for x in g),graphs_with_strict_all_improvement=sum(x['strict_all_improvement']>0 for x in g),**tot))
    csvsave('summary.csv',summary)
    save('asset_binding.json',bound)
    assert all(sha(p)==h for p,h in bound.items())
    production=read(R/'results/joint_output_formal_projected_20261108T043000/source_binding.json');assert all(sha(p)==h for p,h in production.items())
    assert not torch.cuda.is_initialized()
    save('receipt.json',dict(status='complete',graphs=len(rows),full_slot_flips=sum(x['slots'] for x in output if x['method']=='MMD'),selected_slots=len(selected),CPU_threads=1,GPU=0,model_calls=0,updates=0,new_graphs=0,seconds=time.time()-start,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,production_files=len(production),assets_unchanged=True))
    print(json.dumps(read(O/'receipt.json')))
if __name__=='__main__':main()
