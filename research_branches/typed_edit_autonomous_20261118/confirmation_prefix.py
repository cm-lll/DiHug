"""Fresh connected prefixes on one GPU, with independent CPU path validation."""
import subprocess
import confirmation_common as contract
import round22_prefix as prefix
import round30_deploy as prior
from round01 import t,b,np,torch,time,Path,ROOT,SRC,json
OUT=ROOT/'results/typed_edit_autonomous_20261118/confirmation_v1/prefix'
DATASETS=['imdb','pubmed'];SEEDS=[12001,12002];ARMS=['C','D'];CAPS={'imdb':2048,'pubmed':8192}
p=prefix.p;r5=prefix.r5
changes=prefix.changes;independent=prefix.independent;edit=prefix.edit;merging=prefix.merging
prefer_positive_role_coverage=prefix.prefer_positive_role_coverage
mixed_edit_candidate_matrix=prefix.mixed_edit_candidate_matrix
put=lambda name,v:contract.put(OUT,name,v)
verify=lambda:contract.verify(OUT)

def inputs(job):
    folder=Path(job.get('input_folder',OUT/f"inputs/{job['dataset']}/{job['seed']}"))
    return b.restore_graph(b.read(folder/'initial.json')),np.array(b.read(folder/'target.json'))

def stats(ds,g,target):
    v=p.value(ds,g);degree=np.column_stack([g.family_degrees(f) for f in sorted(g.family_specs)])
    return dict(**v,isolates=int((degree.sum(1)==0).sum()),positive_target_zeros=int(((target.sum(1)>0)&(degree.sum(1)==0)).sum()),target_isolates=int((target.sum(1)==0).sum()),degree_error_per_node=float(((degree-target)**2).sum()/g.num_nodes))

@torch.no_grad()
def chain(job,arm,limit,m,device,engineering=False,evaluate=True,deadline=None):
    ds=job['dataset'];seed=job['seed'];base=job.get('base_seed',202611183200000+DATASETS.index(ds)*1000000000+seed*10000);horizon=20000 if ds=='imdb' else 32000;tic=time.monotonic()
    initial,target=inputs(job);state=t.a.CpuAuditState(initial,target,device,use_triton=False);edges,nb=r5.topology(initial);names=sorted(edges)
    gen=torch.Generator(device=device).manual_seed(base+99999);weight=.05;cal={};calls=0
    if arm!='D':
        logits=[];gains=[]
        for _ in range(32):
            batch=state.random_candidates(16,gen,role_phase=True,targeted_fraction=0.,closure_fraction=0.,candidate_mode='legacy_pool')
            logits.append(m(state.compact_features(batch,1.,m.input_dim)[None])[0].cpu());gains.append(state.degree_gains(batch).abs().cpu());calls+=1
        qq=torch.cat(logits);gg=torch.cat(gains);gg=gg[gg>0];std=qq.std().clamp_min(1e-6);q90=torch.quantile(gg,.9).clamp_min(1e-6);weight=float(.5*std*10/q90)
        cal=dict(logit_std=float(std),positive_gain_q90=float(q90),calibration_calls=32)
    trace=[];xx=[];qs=[];arrived=False;snapshot=None;loop_start=time.monotonic();initial_components=p.components(nb)
    for step in range(limit):
        if deadline is not None:assert time.monotonic()<deadline
        draw=base+step;tf=1-step/horizon
        batch=state.random_candidates(16,torch.Generator(device=device).manual_seed(draw),role_phase=True,targeted_fraction=.5,closure_fraction=0.,candidate_mode='legacy_pool')
        vals=t.a.serialize(batch);gain=state.degree_gains(batch).cpu();coverage=state.positive_role_coverage_gains(batch).cpu();x=state.compact_features(batch,tf,m.input_dim)
        if arm=='D':q=torch.zeros(16);score=.05*gain.to(torch.float32)/10+1e-6*torch.rand(16,generator=torch.Generator().manual_seed(draw+5000000))
        else:q=m(x[None])[0].cpu();calls+=1;score=q+weight*gain.to(q.dtype)/10
        if arm!='O':score=prefer_positive_role_coverage(score,gain,coverage)
        merges=merging(edges,nb,names,vals)
        if engineering:assert np.array_equal(merges,merging(edges,nb,names,vals,True))
        if (merges>0).any():score=score+(torch.tensor(merges)>0).to(score.dtype)*((score.max()-score.min()).abs()+1)
        chosen=int(score.argmax());before=p.components(nb);edit(edges,nb,names,vals,chosen);after=p.components(nb);reject=arm!='O' and after>before
        if reject:edit(edges,nb,names,vals,chosen,True);after=p.components(nb)
        effective=0 if reject else chosen;state.commit(batch,torch.tensor(effective,device=device));err=float(state.relation_degree_error());zeros=int(state.positive_role_zero_count())
        if engineering:
            before_graph=r5.graph_from(initial,edges);deg=np.column_stack([before_graph.family_degrees(f) for f in names])
            for u,d in changes(vals,effective).items():deg[u,vals['family'][effective]]-=d
            gg,cg=independent(deg,target,vals);assert np.array_equal(gg,gain.numpy()) and np.array_equal(cg,coverage.numpy())
            assert err==float(((state.family_degree.cpu().numpy().T-target)**2).sum())
        trace.append(dict(step=step+1,proposal_seed=draw,time_fraction=tf,candidates=vals,degree_gains=gain.tolist(),coverage_gains=coverage.tolist(),merge_gains=merges.tolist(),chosen=chosen,effective=effective,rejected=reject,components_before=before,components_after=after,degree_error=err,positive_target_zeros=zeros));xx.append(x.cpu().numpy());qs.append(q.numpy())
        if step+1==1024:
            checkpoint=state.to_graph(initial);snapshot=dict(graph=b.graph_payload(checkpoint),metrics=stats(ds,checkpoint,target))
        if (step+1)%32==0 or step+1==limit:
            if err/initial.num_nodes<=5.5 and (zeros==0 and after==1):arrived=True;break
    loop_seconds=time.monotonic()-loop_start;graph=state.to_graph(initial);graph.validate()
    for f in names:assert edges[f]=={tuple(sorted(e)) for e in graph.family_edges[f]} and len(graph.family_edges[f])==len(initial.family_edges[f])
    assert calls==0 if arm=='D' else calls==32+len(trace)
    report=dict(dataset=ds,seed=seed,arm=arm,steps=len(trace),arrived=arrived,effective_role_weight=weight,calibration=cal,model_calls=calls,initial_components=initial_components,checkpoint_1024=snapshot,seconds=time.monotonic()-tic,loop_seconds=loop_seconds,**(stats(ds,graph,target) if evaluate else dict(components=p.components(nb),degree_error_per_node=err/initial.num_nodes,positive_target_zeros=zeros)))
    return graph,trace,np.stack(xx),np.stack(qs),report

def main():
    b.setup();prior.verify();assert b.read(prior.OUT/'acceptance.json')['passed'] and not b.read(prior.OUT/'summary.json')['replace_F_for_confirmation']
    engineering=ROOT/'results/typed_edit_autonomous_20261118/gpu_prefix_engineering';assert b.read(engineering/'acceptance.json')['passed']
    for f,h in b.read(engineering/'manifest.json')['files'].items():assert b.sha(f)==h,f
    assert torch.cuda.is_available()
    import os
    assert os.environ.get('CUDA_VISIBLE_DEVICES') in [None,'','0,1,2,3,4,5,6,7'],'explicit mapping required for restricted device visibility'
    query=subprocess.run(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout
    inventory=[tuple(int(x.strip()) for x in line.split(',')) for line in query.strip().splitlines()]
    available=[(free,index) for index,free,util in inventory if free>=2048 and util<=5];assert available,'no eligible idle GPU'
    _,card=max(available);device=f'cuda:{card}';torch.cuda.set_device(card);torch.cuda.reset_peak_memory_stats();OUT.mkdir(parents=True,exist_ok=False)
    jobs=[dict(dataset=ds,seed=s) for ds in DATASETS for s in SEEDS]
    for job in jobs:
        ds=job['dataset'];s=job['seed'];ref=b.graph(ds);base=b.MarginalGraphPrior.from_training_graph(ref);role=b.ExchangeableRoleDegreePrior.from_training_graph(ref)
        g=b.FamilyCountCondition(base,role.family_counts(base.family_endpoints)).sample_graph(s+71,mode='binomial');target=role.sample_assignment_for_graph(g,s+91919)
        assert np.array_equal(g.node_types,ref.node_types);put(f'inputs/{ds}/{s}/initial.json',b.graph_payload(g));put(f'inputs/{ds}/{s}/target.json',target.tolist())
    files=[Path(__file__),SRC/'CONFIRMATION_V1.md',prior.OUT/'acceptance.json',prior.OUT/'summary.json',engineering/'acceptance.json']
    files.extend(b.RUN/f'fits/{ds}/seed810/M/model.pt' for ds in DATASETS);files.extend(b.RUN/f'data/{ds}/graph.json' for ds in DATASETS)
    files.extend(OUT/f"inputs/{j['dataset']}/{j['seed']}/{name}" for j in jobs for name in ['initial.json','target.json'])
    contract.bind(OUT,files,jobs=jobs,arms=ARMS,caps=CAPS,device=device,device_name=torch.cuda.get_device_name(card),gpu_inventory=inventory,new_training_updates=0,maximum_scientific_opportunities=40960)
    oldjob=dict(dataset='imdb',seed=108,input_folder=str(prefix.OUT/'inputs/108'),base_seed=2026111821000+100000*108)
    cpu_model=t.a.load_mixed_edit_checkpoint(prefix.CP,'cpu')[0].eval();_,tr,x,q,_=chain(oldjob,'C',16,cpu_model,'cpu',evaluate=False)
    wanted=b.read(prefix.OUT/'outputs/108_C/trace.json')[:16];saved=np.load(prefix.OUT/'outputs/108_C/features.npz')
    assert tr==wanted and np.array_equal(x,saved['x'][:16]);parity=float(abs(q-saved['logits'][:16]).max());assert parity<=2e-5
    models={ds:t.a.load_mixed_edit_checkpoint(b.RUN/f'fits/{ds}/seed810/M/model.pt',device)[0].eval() for ds in DATASETS};cost=30.;eng=[]
    for job in jobs:
        for arm in ARMS:
            chain(job,arm,16,models[job['dataset']],device,True,False)
            _,_,_,_,rp=chain(job,arm,32,models[job['dataset']],device,False,False);cost+=2*(rp['seconds']+(CAPS[job['dataset']]/32-1)*rp['loop_seconds']);eng.append(rp)
    put('engineering.json',dict(passed=cost<=3600,cost_bound_seconds=cost,engineering_opportunities=400,old_cpu_score_error=parity,rows=eng));assert cost<=3600;verify();tic=time.monotonic();rows=[]
    for job in jobs:
        for arm in ARMS:
            g,tr,x,q,rp=chain(job,arm,CAPS[job['dataset']],models[job['dataset']],device,deadline=tic+3600);folder=Path('outputs')/job['dataset']/f"{job['seed']}_{arm}"
            put(folder/'graph.json',b.graph_payload(g));put(folder/'trace.json',tr);put(folder/'report.json',rp);np.savez_compressed(OUT/folder/'features.npz',x=x,logits=q);rows.append(rp)
            print('confirmation prefix generated',job,arm,rp['steps'],rp['arrived'],rp['components'],round(time.monotonic()-tic,1),flush=True)
    peak=torch.cuda.max_memory_allocated();assert peak<=2*1024**3
    put('completion.json',dict(completed=True,new_first_stage_outputs=8,scientific_opportunities=sum(v['steps'] for v in rows),new_training_updates=0,peak_allocated_bytes=peak,seconds=time.monotonic()-tic))
    del models;torch.cuda.empty_cache();tic=time.monotonic();replayed=0;featerr=0.;logerr=0.
    for row in rows:
        ds=row['dataset'];seed=row['seed'];arm=row['arm'];job=dict(dataset=ds,seed=seed);folder=OUT/'outputs'/ds/f'{seed}_{arm}';initial,target=inputs(job);edges,nb=r5.topology(initial);names=sorted(edges);degrees=np.column_stack([initial.family_degrees(f) for f in names]);tr=b.read(folder/'trace.json');saved=np.load(folder/'features.npz')
        if arm=='C':
            m=t.a.load_mixed_edit_checkpoint(b.RUN/f'fits/{ds}/seed810/M/model.pt','cpu')[0].eval();_,q,_=b.c.eval_model(m,saved['x'],np.zeros(len(tr),dtype=np.int64),'cpu');logerr=max(logerr,float(abs(q-saved['logits']).max()));assert logerr<=2e-5
        else:assert not saved['logits'].any() and row['model_calls']==0
        for i,v in enumerate(tr):
            assert time.monotonic()-tic<2400
            vals=v['candidates'];gg,cg=independent(degrees,target,vals);assert np.array_equal(gg,v['degree_gains']) and np.array_equal(cg,v['coverage_gains']);gain=torch.tensor(gg);cov=torch.tensor(cg)
            if i%32==0:
                g=r5.graph_from(initial,edges);fresh=mixed_edit_candidate_matrix(g,t.proposals(g,vals),v['time_fraction'],'typed_closure_compact_v2');featerr=max(featerr,float(abs(fresh-saved['x'][i]).max()));assert featerr<=1e-5
            if arm=='D':score=.05*gain.to(torch.float32)/10+1e-6*torch.rand(16,generator=torch.Generator().manual_seed(v['proposal_seed']+5000000))
            else:score=torch.from_numpy(saved['logits'][i])+row['effective_role_weight']*gain.to(torch.float32)/10
            score=prefer_positive_role_coverage(score,gain,cov);merges=merging(edges,nb,names,vals);assert np.array_equal(merges,v['merge_gains'])
            if (merges>0).any():score=score+(torch.tensor(merges)>0).to(score.dtype)*((score.max()-score.min()).abs()+1)
            assert int(score.argmax())==v['chosen'];before=p.components(nb);assert before==v['components_before'];edit(edges,nb,names,vals,v['chosen']);after=p.components(nb);reject=after>before;assert reject==v['rejected']
            if reject:edit(edges,nb,names,vals,v['chosen'],True);after=p.components(nb)
            effective=0 if reject else v['chosen'];assert effective==v['effective'] and after==v['components_after']
            for u,d in changes(vals,effective).items():degrees[u,vals['family'][effective]]+=d
            assert float(((degrees-target)**2).sum())==v['degree_error'];assert int(((target.sum(1)>0)&(degrees.sum(1)==0)).sum())==v['positive_target_zeros'];replayed+=1
            if i+1==1024:
                g=r5.graph_from(initial,edges);snap=row['checkpoint_1024'];stored=b.restore_graph(snap['graph'])
                for f in names:assert edges[f]=={tuple(sorted(e)) for e in stored.family_edges[f]}
                assert stats(ds,g,target)==snap['metrics']
        final=r5.graph_from(initial,edges);stored=b.restore_graph(b.read(folder/'graph.json'))
        for f in names:assert edges[f]=={tuple(sorted(e)) for e in stored.family_edges[f]} and len(edges[f])==len(initial.family_edges[f])
        for k,v in stats(ds,final,target).items():assert row[k]==v
        assert row['arrived']==(row['degree_error_per_node']<=5.5 and row['positive_target_zeros']==0 and row['components']==1)
        print('confirmation prefix accepted',ds,seed,arm,round(time.monotonic()-tic,1),flush=True)
    put('summary.json',dict(rows=rows,all_C_arrived=all(v['arrived'] for v in rows if v['arm']=='C'),all_D_arrived=all(v['arrived'] for v in rows if v['arm']=='D'),new_original_MLP_fits=0,complete_two_stage=False))
    verify();outputs={str(p):b.sha(p) for p in OUT.rglob('*') if p.is_file() and p.name in ['initial.json','target.json','graph.json','report.json']}
    put('acceptance.json',dict(passed=True,replayed_opportunities=replayed,feature_error=featerr,logit_error=logerr,outputs=outputs,seconds=time.monotonic()-tic))

if __name__=='__main__':
    try:main()
    except BaseException as exc:put('failure.json',dict(type=type(exc).__name__,message=str(exc)));raise
