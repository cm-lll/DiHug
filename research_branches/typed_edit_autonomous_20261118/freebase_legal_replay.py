"""Independent replay of a saved relocation/switch first-stage trajectory."""
import confirmation_prefix_v2 as core
from round01 import t,b,np,torch,time

def check(folder,job,stats,deadline):
    row=b.read(folder/'report.json');tr=b.read(folder/'trace.json');saved=np.load(folder/'features.npz')
    initial,target=core.inputs(job);edges,nb=core.r5.topology(initial);names=sorted(edges)
    degrees=np.column_stack([initial.family_degrees(f) for f in names]);fe=0.;le=0.;tic=time.monotonic()
    ds=job['dataset'];arm=row['arm']
    if arm=='C':
        model=t.a.load_mixed_edit_checkpoint(b.RUN/f'fits/{ds}/seed810/M/model.pt','cpu')[0].eval()
        _,q,_=b.c.eval_model(model,saved['x'],np.zeros(len(tr),dtype=np.int64),'cpu')
        le=float(abs(q-saved['logits']).max());assert le<=2e-5
    else:assert arm=='D' and not saved['logits'].any() and row['model_calls']==0
    for i,v in enumerate(tr):
        assert time.monotonic()<deadline
        vals=v['candidates'];gg,cg=core.independent(degrees,target,vals)
        assert np.array_equal(gg,v['degree_gains']) and np.array_equal(cg,v['coverage_gains'])
        gain=torch.tensor(gg);cov=torch.tensor(cg)
        if i%32==0:
            g=core.r5.graph_from(initial,edges)
            fresh=core.mixed_edit_candidate_matrix(g,t.proposals(g,vals),v['time_fraction'],'typed_closure_compact_v2')
            fe=max(fe,float(abs(fresh-saved['x'][i]).max()));assert fe<=1e-5
        if arm=='D':score=.05*gain.to(torch.float32)/10+1e-6*torch.rand(16,generator=torch.Generator().manual_seed(v['proposal_seed']+5000000))
        else:score=torch.from_numpy(saved['logits'][i])+row['effective_role_weight']*gain.to(torch.float32)/10
        score=core.prefer_positive_role_coverage(score,gain,cov)
        merges=core.merging(edges,nb,names,vals);assert np.array_equal(merges,v['merge_gains'])
        if (merges>0).any():score=score+(torch.tensor(merges)>0).to(score.dtype)*((score.max()-score.min()).abs()+1)
        assert int(score.argmax())==v['unconstrained_chosen']
        before=core.p.components(nb);assert before==v['components_before']
        checked=[]
        for j in sorted(range(16),key=lambda j:(-float(score[j]),j)):
            core.edit(edges,nb,names,vals,j);feasible=core.p.components(nb)<=before;core.edit(edges,nb,names,vals,j,True);checked.append(j)
            if feasible:
                assert j==v['chosen'];break
        else:raise AssertionError('missing admissible noop')
        assert checked==v['legality_checked_indices'] and not v['rejected']
        core.edit(edges,nb,names,vals,v['chosen']);after=core.p.components(nb)
        reject=after>before;assert reject==v['rejected']
        if reject:core.edit(edges,nb,names,vals,v['chosen'],True);after=core.p.components(nb)
        effective=0 if reject else v['chosen'];assert effective==v['effective'] and after==v['components_after']
        for u,d in core.changes(vals,effective).items():degrees[u,vals['family'][effective]]+=d
        assert float(((degrees-target)**2).sum())==v['degree_error']
        assert int(((target.sum(1)>0)&(degrees.sum(1)==0)).sum())==v['positive_target_zeros']
        if i+1==1024:
            g=core.r5.graph_from(initial,edges);snap=row['checkpoint_1024'];stored=b.restore_graph(snap['graph'])
            for f in names:assert edges[f]=={tuple(sorted(e)) for e in stored.family_edges[f]}
            assert stats(ds,g,target)==snap['metrics']
    final=core.r5.graph_from(initial,edges);stored=b.restore_graph(b.read(folder/'graph.json'))
    for f in names:
        assert edges[f]=={tuple(sorted(e)) for e in stored.family_edges[f]}
        assert len(edges[f])==len(initial.family_edges[f])
        assert np.array_equal(final.family_degrees(f),degrees[:,names.index(f)])
    for k,v in stats(ds,final,target).items():assert row[k]==v
    assert len(tr)==row['steps']
    assert row['arrived']==(row['degree_error_per_node']<=5.5 and row['positive_target_zeros']==0 and row['components']==1)
    return dict(passed=True,replayed_opportunities=len(tr),feature_error=fe,logit_error=le,seconds=time.monotonic()-tic)
