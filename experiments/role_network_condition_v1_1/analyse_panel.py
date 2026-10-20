"""Read-only independent CPU audit/plots of the one frozen panel. No model calls."""
import argparse,csv,hashlib,json,math,time
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
import networkx as nx
from scipy.stats import wasserstein_distance

def load(p):return json.loads(p.read_text())
def save(p,obj):p.write_text(json.dumps(obj,indent=2,allow_nan=False))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def avg(x):return float(np.mean(x)) if len(x) else None

def independent(types,edges,schema):
    n=len(types);g=nx.Graph();g.add_nodes_from(range(n));g.add_edges_from(map(tuple,edges))
    assert len(g.edges)==len(edges) and not list(nx.selfloop_edges(g))
    lookup={tuple(pair):i for i,pair in enumerate(schema)};d=np.zeros((n,len(schema)),dtype=int);counts=np.zeros(len(schema),int)
    for u,v in edges:
        assert 0<=u<v<n
        family=lookup[tuple(sorted((int(types[u]),int(types[v]))))];counts[family]+=1;d[u,family]+=1;d[v,family]+=1
    triangles=sum(nx.triangles(g).values())//3
    path_num=path_den=0;pairs=set();closed=set()
    for m in np.flatnonzero(types==2):
        for u in [v for v in g[m] if types[v]==0]:
            for v in [v for v in g[m] if types[v]==1]:
                path_den+=1;pairs.add((u,v))
                if g.has_edge(u,v):path_num+=1;closed.add((u,v))
    return dict(graph=g,degree=d,family_edges=counts.tolist(),components=sorted(map(len,nx.connected_components(g)),reverse=True),
        triangles=int(triangles),isolates=len(list(nx.isolates(g))),path_num=path_num,path_den=path_den,
        pair_num=len(closed),pair_den=len(pairs),joint={t:Counter(map(tuple,d[types==t])) for t in range(4)})

def mp_independent(types,edges,schema):
    # Independent depth-first ACTUAL simple paths, unlike production category-product counts.
    adj=[{} for _ in types];lookup={tuple(p):r for r,p in enumerate(schema)}
    for u,v in edges:
        r=lookup[tuple(sorted((int(types[u]),int(types[v]))))];adj[u][int(v)]=r;adj[v][int(u)]=r
    result=[Counter(),Counter(),Counter()]
    def visit(seq,pattern):
        if len(seq)>1 and seq[0]<seq[-1]:
            result[len(seq)-2][','.join(map(str,min(pattern,pattern[::-1])))]+=1
        if len(seq)==4:return
        for v,r in adj[seq[-1]].items():
            if v not in seq:visit(seq+[v],pattern+(r,int(types[v])))
    for u in range(len(types)):visit([u],(int(types[u]),))
    return result

def main(run):
    start=time.monotonic();run=Path(run).resolve();ledger=load(run.with_name(run.name+'.budget.json'))
    cutoff=ledger['hard_cutoff_epoch'];out=run/'analysis';out.mkdir(exist_ok=False);status='incomplete';reason=None
    def check():
        if time.time()>=cutoff or (run/'STOP').exists():raise TimeoutError('original cumulative budget/STOP; never renewed')
    try:
        check();cfg=load(run/'resolved_config.json');panel=load(run/'panel.json');schema=panel['schema'];types=np.asarray(panel['graphs'][0]['types'])
        refs={c:independent(types,np.asarray(next(x for x in panel['graphs'] if x['condition']==c)['edges']),schema) for c in ['balanced','extreme']}
        state=load(run/'summary.json');rows=[];numeric=[];graphs={};partial=[]
        for seed in cfg['train_seeds']:
            p=run/('seed'+str(seed))/'generation.json'
            if p.exists():rows.extend(load(p))
            else:partial.append(seed)
        # Recompute every saved binary graph, including graphs whose enclosing seed was interrupted.
        for p in sorted(run.glob('seed*/*.npz')):
            check();data=np.load(p,allow_pickle=False);assert np.array_equal(data['types'],types) and np.array_equal(data['schema'],schema)
            edges=np.asarray(data['edges']);ind=independent(types,edges,schema)
            rel=np.asarray([schema.index(sorted([int(types[u]),int(types[v])]))+1 for u,v in edges]);assert np.array_equal(data['relations'],rel)
            graphs[str(p.relative_to(run))]=ind
            for r,count in enumerate(panel['graphs'][0]['family_counts']):
                if count==0:assert ind['family_edges'][r]==0
        mp_audited=0;max_w1_error=0.;row_errors=[];summary_rows=[]
        for row in rows:
            check();seed,arm,condition,ss=[row[k] for k in ['seed','arm','condition','sampling_seed']]
            stored_condition=condition if arm=='C' else 'balanced';name=f'seed{seed}/{arm}_{stored_condition}_{ss}.npz'
            ind=graphs[name];g=ind['graph'];d=ind['degree'];ref=refs[condition];m=row['metrics']
            assert ind['family_edges']==m['family_edges'] and sum(ind['family_edges'])==m['edges']
            assert ind['components']==m['component_sizes'] and ind['isolates']==m['isolates'] and ind['triangles']==m['triangles']
            closure=m['imdb_closure'];assert (ind['path_num'],ind['path_den'],ind['pair_num'],ind['pair_den'])==(closure['path_numerator'],closure['path_denominator'],closure['unique_pair_numerator'],closure['unique_pair_denominator'])
            distances={};jtv={}
            for t,typ in enumerate(panel['type_names']):
                ix=types==t;distances[typ]=[float(wasserstein_distance(d[ix,r],ref['degree'][ix,r])) for r in range(len(schema))]
                for a,b in zip(distances[typ],row['comparison']['typed_relation_degree_W1'][typ]):max_w1_error=max(max_w1_error,abs(a-b));assert abs(a-b)<1e-10
                ra,rb=ind['joint'][t],ref['joint'][t];n=int(ix.sum());jtv[typ]=.5*sum(abs(ra[v]-rb[v])/n for v in ra.keys()|rb.keys())
                assert abs(jtv[typ]-row['comparison']['joint_relation_degree_TV'][typ])<1e-12
            # Fixed first sampling ID, every actual model/condition; avoid duplicated A/B evaluations.
            if ss==cfg['sampling_seeds'][0] and (arm=='C' or condition=='balanced'):
                data=np.load(run/name,allow_pickle=False);counts=mp_independent(types,data['edges'],schema)
                assert [dict(c) for c in counts]==[x['counts'] for x in m['mp']];mp_audited+=1
            gd=d[types==3,8];high=int((gd>=cfg['G_high_degree_threshold']).sum());assert high==row['G_high_ge12']
            item=dict(seed=seed,arm=arm,condition=condition,sampling_seed=ss,raw_path=name,
                edges=m['edges'],family_edges=m['family_edges'],family_abs_error=np.abs(np.array(ind['family_edges'])-ref['family_edges']).tolist(),
                typed_relation_W1=distances,joint_TV=jtv,G_W1=distances['Genre'][8],G_joint_TV=jtv['Genre'],G_degrees=gd.tolist(),
                G_max=int(gd.max()),G_high_ge12=high,multiple_hubs=high>=2,all_six_hubs=high==6,
                G_low_le1=int((gd<=1).sum()),G_zero=int((gd==0).sum()),
                connected=nx.is_connected(g),components=len(ind['components']),LCC=ind['components'][0],isolates=ind['isolates'],
                triangles=ind['triangles'],path_closed=ind['path_num'],path_total=ind['path_den'],path_closure=ind['path_num']/ind['path_den'] if ind['path_den'] else None,
                pair_closed=ind['pair_num'],pair_total=ind['pair_den'],pair_closure=ind['pair_num']/ind['pair_den'] if ind['pair_den'] else None,
                topology_lower_bound_key=row['topology_invariant'],graph_sha256=row['raw_graph_sha256'])
            numeric.append(item)
        updates=[json.loads(s) for s in (run/'updates.jsonl').read_text().splitlines()] if (run/'updates.jsonl').exists() else []
        paired=defaultdict(dict)
        for row in updates:paired[(row['seed'],row['update'])][row['arm']]=row
        complete_update_triples=0
        for key,arms in paired.items():
            if len(arms)==3:
                assert len({x['input_sha256'] for x in arms.values()})==1;complete_update_triples+=1
        validations=[]
        for p in sorted(run.glob('seed*/*validation_*.json')):
            v=load(p);v['positive_ce']=sum(sum(r['ce_sum_by_family_label'][j][1] for j in range(len(schema))) for r in v['rows'])/sum(sum(r['counts_by_family_label'][j][1] for j in range(len(schema))) for r in v['rows'])
            v['negative_ce']=sum(sum(r['ce_sum_by_family_label'][j][0] for j in range(len(schema))) for r in v['rows'])/sum(sum(r['counts_by_family_label'][j][0] for j in range(len(schema))) for r in v['rows']);validations.append(v)
        fields=['G_W1','G_joint_TV','G_max','G_high_ge12','multiple_hubs','G_low_le1','G_zero','edges','connected','components','LCC','isolates','triangles','path_closure','pair_closure']
        for seed in cfg['train_seeds']:
            for condition in ['balanced','extreme']:
                for arm in ['A','B','C']:
                    selected=[r for r in numeric if (r['seed'],r['condition'],r['arm'])==(seed,condition,arm)]
                    entry=dict(seed=seed,condition=condition,arm=arm,n=len(selected),planned_n=8,
                        metrics={f:dict(mean=avg([float(r[f]) for r in selected if r[f] is not None]),median=float(np.median([r[f] for r in selected if r[f] is not None])) if selected and any(r[f] is not None for r in selected) else None,p90=float(np.quantile([r[f] for r in selected if r[f] is not None],.9)) if selected and any(r[f] is not None for r in selected) else None,NA=sum(r[f] is None for r in selected)) for f in fields},
                        provably_distinct_topologies_lower_bound=len({r['topology_lower_bound_key'] for r in selected}))
                    if selected:
                        entry['mean_family_edges']=np.mean([r['family_edges'] for r in selected],axis=0).tolist()
                        entry['mean_family_absolute_error']=np.mean([r['family_abs_error'] for r in selected],axis=0).tolist()
                        entry['joint_TV_by_type']={t:avg([r['joint_TV'][t] for r in selected]) for t in panel['type_names']}
                        entry['W1_by_type']={t:np.mean([r['typed_relation_W1'][t] for r in selected],axis=0).tolist() for t in panel['type_names']}
                    summary_rows.append(entry)
        macro=[];contrasts=[]
        for condition in ['balanced','extreme']:
            for arm in ['A','B','C']:
                cells=[x for x in summary_rows if x['condition']==condition and x['arm']==arm and x['n']==8]
                macro.append(dict(condition=condition,arm=arm,training_seeds=len(cells),means={f:avg([x['metrics'][f]['mean'] for x in cells if x['metrics'][f]['mean'] is not None]) for f in fields},
                    mean_family_edges=np.mean([x['mean_family_edges'] for x in cells],axis=0).tolist() if cells else None,
                    joint_TV_by_type={t:avg([x['joint_TV_by_type'][t] for x in cells]) for t in panel['type_names']}))
            for left,right in [('B','A'),('C','B'),('C','A')]:
                for f in fields:
                    diffs=[]
                    for seed in cfg['train_seeds']:
                        a=next(x for x in summary_rows if (x['seed'],x['condition'],x['arm'])==(seed,condition,left));b=next(x for x in summary_rows if (x['seed'],x['condition'],x['arm'])==(seed,condition,right))
                        if a['n']==b['n']==8 and a['metrics'][f]['mean'] is not None and b['metrics'][f]['mean'] is not None:diffs.append(dict(seed=seed,difference=a['metrics'][f]['mean']-b['metrics'][f]['mean']))
                    contrasts.append(dict(condition=condition,contrast=left+'-'+right,metric=f,by_training_seed=diffs,mean_difference=avg([d['difference'] for d in diffs]),scope='descriptive paired training-seed means; n=3, no independent-training inference from8 draws'))
        check();save(out/'raw_graph_metrics.json',numeric);save(out/'seed_summary.json',summary_rows);save(out/'macro_summary.json',macro);save(out/'paired_differences.json',contrasts);save(out/'validation_curve.json',validations)
        conditional=[]
        for seed in cfg['train_seeds']:
            paired_graphs=[]
            for ss in cfg['sampling_seeds']:
                a=next((r for r in rows if (r['seed'],r['arm'],r['condition'],r['sampling_seed'])==(seed,'C','balanced',ss)),None)
                b=next((r for r in rows if (r['seed'],r['arm'],r['condition'],r['sampling_seed'])==(seed,'C','extreme',ss)),None)
                if a and b:
                    assert a['initial_state_sha256']==b['initial_state_sha256']
                    aa=np.load(run/f'seed{seed}/C_balanced_{ss}.npz',allow_pickle=False)['edges'];bb=np.load(run/f'seed{seed}/C_extreme_{ss}.npz',allow_pickle=False)['edges']
                    distance=len(set(map(tuple,aa))^set(map(tuple,bb)))
                    paired_graphs.append(dict(sampling_seed=ss,same_initial_noise=True,labeled_edge_symmetric_difference=distance,
                        balanced_G_high=a['G_high_ge12'],extreme_G_high=b['G_high_ge12'],balanced_G_max=a['G_max'],extreme_G_max=b['G_max']))
            lp=run/f'seed{seed}/actual_condition_response.json'
            conditional.append(dict(seed=seed,actual_paired_graphs=paired_graphs,local_fixed_input_response=load(lp) if lp.exists() else None,
                caution='labeled graph differences are not anonymous topology classes; nonzero p0 response is not quality improvement'))
        save(out/'conditional_response.json',conditional)
        with (out/'training_curve.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(updates[0]) if updates else ['no_updates']);writer.writeheader();writer.writerows(updates)
        import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
        colors={'A':'#4266a9','B':'#d18728','C':'#2b8b64'};fig,axes=plt.subplots(3,2,figsize=(12,10))
        for i,seed in enumerate(cfg['train_seeds']):
            for arm in ['A','B','C']:
                u=[r for r in updates if r['seed']==seed and r['arm']==arm]
                if u:
                    y=np.array([r['loss'] for r in u]);x=np.array([r['update'] for r in u]);axes[i,0].plot(x,y,color=colors[arm],alpha=.15,linewidth=.5)
                    if len(y)>=50:axes[i,0].plot(x[49:],np.convolve(y,np.ones(50)/50,'valid'),color=colors[arm],label=arm)
                v=sorted([r for r in validations if r['seed']==seed and r['arm']==arm],key=lambda r:r['update'])
                axes[i,1].plot([r['update'] for r in v],[r['mean_ce'] for r in v],'-o',color=colors[arm],label=arm)
            axes[i,0].set_title(f'Seed {seed}: train CE,50-update moving mean');axes[i,1].set_title('Fixed full-panel validation CE')
            for ax in axes[i]:ax.set_xlabel('Effective update');ax.set_ylabel('CE');ax.legend();ax.grid(alpha=.2)
        check();fig.tight_layout();fig.savefig(out/'learning_curves.png',dpi=150);fig.savefig(out/'learning_curves.svg');plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for ax,condition in zip(axes,['balanced','extreme']):
            target=[0]*19;target[4]=1 if condition=='balanced' else 0
            if condition=='extreme':target[1]=4/6;target[2]=1/6;target[18]=1/6
            ax.plot(range(19),target,'k--',label='Target')
            for arm in ['A','B','C']:
                pmfs=[]
                for seed in cfg['train_seeds']:
                    rs=[r for r in numeric if (r['seed'],r['arm'],r['condition'])==(seed,arm,condition)]
                    if len(rs)==8:pmfs.append(np.mean([np.bincount(r['G_degrees'],minlength=19)/6 for r in rs],axis=0))
                if pmfs:ax.plot(range(19),np.mean(pmfs,axis=0),'-o',markersize=3,label=arm,color=colors[arm])
            ax.set_title(condition+' Genre: actual sampled degree PMF');ax.set_xlabel('Degree');ax.set_ylabel('Probability');ax.legend();ax.grid(alpha=.2)
        check();fig.tight_layout();fig.savefig(out/'sampled_Genre_degrees.png',dpi=150);fig.savefig(out/'sampled_Genre_degrees.svg');plt.close(fig)
        graph_hashes={name:digest(run/name) for name in graphs};save(out/'raw_graph_hashes.json',graph_hashes)
        save(out/'weight_hashes.json',{str(p.relative_to(run)):digest(p) for p in run.glob('seed*/*.pt')})
        assert len(numeric)==len(rows)
        report=dict(worker_status=state['status'],completed_rows=len(rows),actual_saved_graphs=len(graphs),planned_actual_graphs=96,
            planned_evaluation_rows=144,updates=len(updates),planned_updates=9000,complete_paired_update_triples=complete_update_triples,
            independent_all_saved_graph_counts_degrees_components_triangles_closure=True,independent_all_recorded_W1_joint_TV=True,
            independently_enumerated_MP_graphs=mp_audited,max_W1_discrepancy=max_w1_error,incomplete_seed_records=partial,
            scientific_inputs_unchanged=True,sampling_requests_independent_training_seeds=False,new_models_or_samples=0)
        save(out/'audit.json',report);status='completed';return 0
    except BaseException as e:reason=repr(e);raise
    finally:save(out/'analysis_status.json',dict(status=status,reason=reason,seconds=time.monotonic()-start,original_cutoff_epoch=cutoff,budget_reset=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();raise SystemExit(main(a.run))
