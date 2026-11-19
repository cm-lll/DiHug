"""Independent CPU finalization of fully saved prefixes after worker termination."""
import confirmation_common as contract
import confirmation_prefix_v2 as producer
from round01 import t,b,np,torch,time,Path,ROOT,SRC,json
OUT=producer.OUT
FINAL=ROOT/'results/typed_edit_autonomous_20261118/confirmation_v1/prefix_finalized'
r5=producer.r5;p=producer.p;stats=producer.stats;inputs=producer.inputs;changes=producer.changes
independent=producer.independent;merging=producer.merging;edit=producer.edit
mixed_edit_candidate_matrix=producer.mixed_edit_candidate_matrix
prefer_positive_role_coverage=producer.prefer_positive_role_coverage

def main():
    b.setup();assert not (OUT/'acceptance.json').exists();assert b.read(OUT/'completion.json')['new_first_stage_outputs']==8
    for f,h in b.read(OUT/'manifest.json')['files'].items():assert b.sha(f)==h,f
    FINAL.mkdir(parents=True,exist_ok=False)
    contract.put(OUT,'interruption.json',dict(exit_code=143,phase='independent_acceptance',scientific_completion_saved=True,cause='external termination; exact reason unconfirmed',new_graphs_for_recovery=0))
    files=[Path(__file__),SRC/'CONFIRMATION_PREFIX_FINALIZE.md',OUT/'manifest.json',OUT/'completion.json',OUT/'interruption.json']
    files.extend(q for q in OUT.rglob('*') if q.is_file())
    contract.bind(FINAL,files,new_training_updates=0,new_graphs=0,source=str(OUT))
    rows=[b.read(OUT/f'outputs/{ds}/{sid}_{arm}/report.json') for ds in producer.DATASETS for sid in producer.SEEDS for arm in producer.ARMS]
    tic=time.monotonic();replayed=0;featerr=0.;logerr=0.
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
    summary=dict(rows=rows,all_C_arrived=all(v['arrived'] for v in rows if v['arm']=='C'),all_D_arrived=all(v['arrived'] for v in rows if v['arm']=='D'),new_original_MLP_fits=0,complete_two_stage=False,independent_finalization=str(FINAL))
    contract.put(FINAL,'summary.json',summary);contract.verify(FINAL)
    outputs={str(q):b.sha(q) for q in OUT.rglob('*') if q.is_file() and q.name in ['initial.json','target.json','graph.json','report.json']}
    receipt=dict(passed=True,replayed_opportunities=replayed,feature_error=featerr,logit_error=logerr,outputs=outputs,seconds=time.monotonic()-tic,new_graphs=0,new_training_updates=0)
    assert replayed==b.read(OUT/'completion.json')['scientific_opportunities']
    contract.put(FINAL,'acceptance.json',receipt)
    delegated=dict(receipt,acceptance_source=str(FINAL/'acceptance.json'),acceptance_source_sha256=b.sha(FINAL/'acceptance.json'),independent_finalization=True,original_worker_exit_code=143)
    delegated['outputs']={**outputs,**{str(q):b.sha(q) for q in [Path(__file__),SRC/'CONFIRMATION_PREFIX_FINALIZE.md',FINAL/'manifest.json',FINAL/'acceptance.json',FINAL/'summary.json']}}
    assert not (OUT/'summary.json').exists() and not (OUT/'acceptance.json').exists()
    contract.put(OUT,'summary.json',summary);contract.put(OUT,'acceptance.json',delegated)
    print('All eight saved prefixes independently finalized',replayed,flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as exc:contract.put(FINAL,'failure.json',dict(type=type(exc).__name__,message=str(exc)));raise
