import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from worker import *
import csv
def write(name,rows):
    if not rows:return
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    start=time.time();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    protocol=json.loads((OUT/'protocol.json').read_text());rows=[];relations=[];nodes=[];checks=[];models=[]
    old_table=list(csv.DictReader((OLD/'evaluation_summary.csv').open()))
    for row in protocol['tasks']:
        ds,seed=row['dataset'],row['seed'];folder=OUT/f'{ds}_{seed}';old=OLD/folder.name
        if not (folder/'checks.json').exists():continue
        update=json.loads((folder/'updates.json').read_text())[0];train=torch.load(folder/'update_input.pt',map_location='cpu',weights_only=False);fixed=torch.load(old/'evaluation_inputs.pt',map_location='cpu',weights_only=False)['inputs']
        data=np.load(row['data']);inp=fixed[0];ix=np.argsort(inp['ids']);core=HardScore(data['types'],inp['query'][:,ix],inp['f'][ix],inp['labels'][ix],len(data['schema']));exact=Exact(core)
        outputs=json.loads((folder/'evaluation.json').read_text());modelrows=[]
        for label,inp,j in [('same',train,-1)]+[(f't{x["t"]}',x,j) for j,x in enumerate(fixed)]:
            before=np.load(folder/'same_before.npz' if label=='same' else old/f'eval_before_t{inp["t"]}.npz');after=np.load(folder/f'{label}_after.npz')
            for name in ('ids','labels','families','query'):assert np.array_equal(before[name],after[name])
            assert next(x for x in outputs if x['panel']==label)['input_hash']==rt.input_hash([inp])
            eval_seed=seed+protocol['same_input_evaluation_seed_offset'] if label=='same' else seed+protocol['evaluation_hard_seed_offset']+j
            uniform=np.random.default_rng(eval_seed).random((256,len(core.f)));scored={}
            for arm,z in [('before',before),('after',after)]:
                p=z['probability'];terms=z['terms'];expected=np.zeros((core.n,core.r));np.add.at(expected,(core.q[0],core.f),p);np.add.at(expected,(core.q[1],core.f),p);ref=core.ref['degree']
                if arm=='before' and label!='same':
                    cache=np.load(old/f'evaluation_scored_before_t{inp["t"]}.npz');values=cache['values'];costs=cache['costs_remaining'];orig=next(x for x in old_table if x['dataset']==ds and int(x['seed'])==seed and x['arm']=='before' and int(x['t'])==inp['t']);tri=float(orig['expected_triangles'])
                else:
                    V,_,stats=exact.value_gradient(p);costs=remaining(core,(uniform<p).astype(np.uint8));values=np.zeros(5);values[[0,2,3]]=V;values[[1,4]]=costs.mean(0);tri=sum(x['mean'] for x in stats['triangles'])
                scored[arm]=(values,costs)
                np.savez_compressed(folder/f'{label}_{arm}_scored.npz',values=values,costs_remaining=costs,expected_degree=expected)
                meta=dict(dataset=ds,seed=seed,panel=label,t=inp['t'],arm=arm)
                total=float(terms.sum()*inp['ce_weight']);pos=core.reference==1;neg=~pos
                modelrows.append(dict(**meta,**{n:float(v) for n,v in zip(core.names,values)},mean_five=float(values.mean()),CE=total,positive_CE=float(terms[pos].mean()),negative_CE=float(terms[neg].mean()),expected_edges=float(p.sum()),expected_triangles=tri,node_total_degree_MAE=float(abs(expected.sum(1)-ref.sum(1)).mean()),node_relation_degree_MAE=float(abs(expected-ref).mean())))
                for f in range(core.r):
                    m=core.f==f;pos=m&(core.reference==1);neg=m&(core.reference==0)
                    relations.append(dict(**meta,family=f,support=int(m.sum()),positive_n=int(pos.sum()),negative_n=int(neg.sum()),reference_edges=int(core.ref['counts'][f]),expected_edges=float(p[m].sum()),positive_missed_mass=float((1-p[pos]).sum()),negative_false_mass=float(p[neg].sum()),positive_CE=float(terms[pos].mean()) if pos.any() else None,negative_CE=float(terms[neg].mean()) if neg.any() else None))
                d=ref.sum(1)
                for ty in np.unique(core.types):
                    m=core.types==ty;cut=float(np.quantile(d[m],.9));hub=m&(d>=cut);low=m&(d<=2)&~hub;mid=m&~hub&~low
                    for group,selected in [('low',low),('middle',mid),('hub',hub)]:
                        n=int(selected.sum());nodes.append(dict(**meta,node_type=int(ty),group=group,n=n,reference_hub_threshold=cut,total_degree_MAE=float(abs(expected.sum(1)-d)[selected].mean()) if n else None,relation_degree_MAE=float(abs(expected-ref)[selected].mean()) if n else None))
            delta=scored['after'][0]-scored['before'][0];samplediff=scored['after'][1]-scored['before'][1]
            for k,n in enumerate(core.names):
                slope=float(update['actual_directional_derivatives'][k]) if label=='same' else None
                rows.append(dict(dataset=ds,seed=seed,panel=label,t=inp['t'],metric=n,analytic=k in (0,2,3),before=float(scored['before'][0][k]),after=float(scored['after'][0][k]),difference=float(delta[k]),relative_change=float(delta[k]/scored['before'][0][k]) if scored['before'][0][k]>1e-12 else None,paired_MC_SE=float(samplediff[:,[1,4].index(k)].std(ddof=1)/16) if k in (1,4) else 0.,first_order_slope=slope,finite_minus_linear=float(delta[k]-slope) if slope is not None else None,evaluation_seed=eval_seed))
        models+=modelrows
        z=np.load(folder/'train_displacement_u2001.npz');p=z['probability'];V,G,_=exact.value_gradient(p);cost=remaining(core,z['bits']);feedback=np.zeros_like(z['feedback']);feedback[[0,2,3]]=G;feedback[[1,4]]=loo(p,z['bits'],cost);np.testing.assert_allclose(feedback,z['feedback'],rtol=1e-10,atol=1e-12)
        lam,cert=solver(z['Gram'],z['proposal_inner_products']);np.testing.assert_allclose(lam,z['coefficients'],rtol=1e-9,atol=1e-9);assert max(update['actual_update_cosines'])<=1e-5
        final=torch.load(folder/'displacement_complete.pt',map_location='cpu',weights_only=False);parent=torch.load(row['checkpoint'],map_location='cpu',weights_only=False)
        assert final['state']['updates']==final['state']['ema']['updates']==2001
        assert g.digest(final['state']['rng'])==update['main_rng_hash'];assert g.digest(final['auxiliary_rng'])==update['aux_after_hash']
        assert rt.input_hash([train])==row['diagnostic_input_hash']==update['input_hash']
        for pid,state in final['state']['optimizer']['state'].items():assert float(state['step'])-float(parent['optimizer']['state'][pid]['step']) in (0,1)
        checks.append(dict(dataset=ds,seed=seed,same_input_hash=update['input_hash'],actual_written_constraint_max_cos=max(update['actual_update_cosines']),feedback_max_error=float(abs(feedback-z['feedback']).max()),solver_residual=cert['residual'],full_state_and_RNG_checked=True))
    for p,h in json.loads((OUT/'bindings.json').read_text()).items():assert rt.sha(p)==h,p
    write('risk_changes.csv',rows);write('evaluation_summary.csv',models);write('relations.csv',relations);write('node_groups.csv',nodes)
    rt.save(OUT/'audit.json',dict(passed=True,models=checks,seconds=time.time()-start,GPU_hours=0,model_calls=0,all_bound_files_unchanged=True))
    summaries=[];tol=protocol['risk_difference_reporting_tolerance']
    for row in protocol['tasks']:
        ds,seed=row['dataset'],row['seed'];rr=[r for r in rows if r['dataset']==ds and r['seed']==seed];same=[r for r in rr if r['panel']=='same'];other=[r for r in rr if r['panel']!='same']
        if not same:continue
        summaries.append(dict(dataset=ds,seed=seed,train_t=row['diagnostic_t'],same_before=float(np.mean([r['before'] for r in same])),same_after=float(np.mean([r['after'] for r in same])),same_analytic_harm=[r['metric'] for r in same if r['analytic'] and r['difference']>tol],same_details=same,other_times=[dict(t=t,before=float(np.mean([r['before'] for r in other if r['t']==t])),after=float(np.mean([r['after'] for r in other if r['t']==t])),analytic_harm=[r['metric'] for r in other if r['t']==t and r['analytic'] and r['difference']>tol]) for t in protocol['evaluation_times']]))
    rt.save(OUT/'decision_summary.json',dict(models=summaries,interpretation='Local one-step diagnostic; exact same-input finite-minus-linear residual does not uniquely identify curvature or prescribe a step size. MC terms and cached other-input baselines carry separate limits.'))
    print(json.dumps(summaries,indent=2))
if __name__=='__main__':main()

