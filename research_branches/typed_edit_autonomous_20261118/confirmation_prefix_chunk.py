"""Independent CPU finalization of fully saved prefixes after worker termination."""
import argparse
import confirmation_common as contract
import confirmation_prefix_v2 as producer
from round01 import t,b,np,torch,time,Path,ROOT,SRC,json
OUT=producer.OUT
CHUNKS=ROOT/'results/typed_edit_autonomous_20261118/confirmation_v1/prefix_chunks'
FINAL=None
r5=producer.r5;p=producer.p;stats=producer.stats;inputs=producer.inputs;changes=producer.changes
independent=producer.independent;merging=producer.merging;edit=producer.edit
mixed_edit_candidate_matrix=producer.mixed_edit_candidate_matrix
prefer_positive_role_coverage=producer.prefer_positive_role_coverage

def main(index):
    global FINAL
    b.setup();FINAL=CHUNKS/str(index);FINAL.mkdir(parents=True,exist_ok=False)
    for f,h in b.read(OUT/'manifest.json')['files'].items():assert b.sha(f)==h,f
    jobs=[(ds,sid,arm) for ds in producer.DATASETS for sid in producer.SEEDS for arm in producer.ARMS];ds,sid,arm=jobs[index]
    folder=OUT/f'outputs/{ds}/{sid}_{arm}';rows=[b.read(folder/'report.json')]
    files=[Path(__file__),SRC/'CONFIRMATION_PREFIX_CHUNKS.md',OUT/'manifest.json',OUT/'completion.json']
    files.extend(folder/name for name in ['graph.json','report.json','trace.json','features.npz'])
    files.extend(OUT/f'inputs/{ds}/{sid}/{name}' for name in ['initial.json','target.json'])
    files.extend([b.RUN/f'fits/{ds}/seed810/M/model.pt',b.RUN/f'data/{ds}/graph.json'])
    contract.bind(FINAL,files,index=index,dataset=ds,seed=sid,arm=arm,new_graphs=0,new_training_updates=0)
    tic=time.monotonic();replayed=0;featerr=0.;logerr=0.
    for row in rows:
        ds=row['dataset'];seed=row['seed'];arm=row['arm'];job=dict(dataset=ds,seed=seed);folder=OUT/'outputs'/ds/f'{seed}_{arm}';initial,target=inputs(job);edges,nb=r5.topology(initial);names=sorted(edges);degrees=np.column_stack([initial.family_degrees(f) for f in names]);tr=b.read(folder/'trace.json');saved=np.load(folder/'features.npz')
        if arm=='C':
            m=t.a.load_mixed_edit_checkpoint(b.RUN/f'fits/{ds}/seed810/M/model.pt','cpu')[0].eval();_,q,_=b.c.eval_model(m,saved['x'],np.zeros(len(tr),dtype=np.int64),'cpu');logerr=max(logerr,float(abs(q-saved['logits']).max()));assert logerr<=2e-5
        else:assert not saved['logits'].any() and row['model_calls']==0
        for i,v in enumerate(tr):
            assert time.monotonic()-tic<300
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
    assert replayed==rows[0]['steps'];contract.verify(FINAL)
    contract.put(FINAL,'acceptance.json',dict(passed=True,index=index,dataset=ds,seed=sid,arm=arm,replayed_opportunities=replayed,feature_error=featerr,logit_error=logerr,seconds=time.monotonic()-tic,new_graphs=0,new_training_updates=0))
    print('Prefix chunk finalized',index,replayed,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('index',type=int,choices=range(8));args=parser.parse_args()
    try:main(args.index)
    except BaseException as exc:
        if FINAL is not None:contract.put(FINAL,'failure.json',dict(type=type(exc).__name__,message=str(exc)))
        raise
