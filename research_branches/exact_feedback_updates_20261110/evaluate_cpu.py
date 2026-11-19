"""Post-GPU evaluation of saved conditional clean probabilities; no model calls."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from worker import *
import csv
def write(name,rows):
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    start=time.time();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    protocol=json.loads((OUT/'protocol.json').read_text());sup=json.loads((OUT/'supervisor.json').read_text())
    results=[];relations=[];nodes=[];pairs=[];summaries=[];checks=[];allcases={}
    for task in protocol['tasks']:
        ds,seed=task['dataset'],task['seed'];folder=OUT/f'{ds}_{seed}'
        if not (folder/'receipt.json').exists() or json.loads((folder/'receipt.json').read_text())['status']!='completed':continue
        data=np.load(task['data']);inputs=torch.load(folder/'evaluation_inputs.pt',map_location='cpu',weights_only=False)
        inp=inputs['inputs'][0];ix=np.argsort(inp['ids']);core=HardScore(data['types'],inp['query'][:,ix],inp['f'][ix],inp['labels'][ix],len(data['schema']));exact=Exact(core)
        ev=json.loads((folder/'evaluation.json').read_text());updates=json.loads((folder/'updates.json').read_text());values={};costs2={}
        for j,t in enumerate(protocol['evaluation_times']):
            uniform=np.random.default_rng(seed+protocol['evaluation_hard_seed_offset']+j).random((protocol['evaluation_hard_draws'],len(core.reference)))
            for arm in ('before',*protocol['arms']):
                pred=np.load(folder/f'eval_{arm}_t{t}.npz');p=pred['probability'];V,G,stats=exact.value_gradient(p)
                bits=(uniform<p).astype(np.uint8);C=remaining(core,bits);val=np.zeros(5);val[[0,2,3]]=V;val[[1,4]]=C.mean(0);values[arm,t]=val;costs2[arm,t]=C
                info=next(x for x in ev if x['arm']==arm and x['t']==t)
                row=dict(dataset=ds,seed=seed,arm=arm,t=t,**{n:float(v) for n,v in zip(core.names,val)},mean_five=float(val.mean()),CE=info['CE'],positive_CE=info['positive_CE'],negative_CE=info['negative_CE'],expected_edges=float(p.sum()),expected_triangles=sum(x['mean'] for x in stats['triangles']))
                expected=np.zeros((core.n,core.r));np.add.at(expected,(core.q[0],core.f),p);np.add.at(expected,(core.q[1],core.f),p)
                ref=core.ref['degree'];row['node_total_degree_MAE']=float(abs(expected.sum(1)-ref.sum(1)).mean());row['node_relation_degree_MAE']=float(abs(expected-ref).mean())
                results.append(row);allcases[ds,seed,arm,t]=row
                for f in range(core.r):
                    mask=core.f==f;pos=mask&(core.reference==1);neg=mask&(core.reference==0);ii=next(x for x in info['relations'] if x['family']==f)
                    relations.append(dict(dataset=ds,seed=seed,arm=arm,t=t,family=f,support=int(mask.sum()),reference_edges=int(core.ref['counts'][f]),expected_edges=float(p[mask].sum()),missed_positive_mass=float((1-p[pos]).sum()),false_positive_mass=float(p[neg].sum()),positive_CE=ii['positive_CE'],negative_CE=ii['negative_CE']))
                for ty in np.unique(core.types):
                    mask=core.types==ty;d=ref.sum(1);cut=float(np.quantile(d[mask],.9));hub=mask&(d>=cut);low=mask&(d<=2)&~hub;mid=mask&~hub&~low
                    for label,selected in [('low',low),('middle',mid),('hub',hub)]:
                        n=int(selected.sum());nodes.append(dict(dataset=ds,seed=seed,arm=arm,t=t,node_type=int(ty),group=label,n=n,reference_hub_threshold=cut,total_degree_MAE=float(abs(expected.sum(1)-d)[selected].mean()) if n else None,relation_degree_MAE=float(abs(expected-ref)[selected].mean()) if n else None,mean_expected_degree=float(expected.sum(1)[selected].mean()) if n else None,mean_reference_degree=float(d[selected].mean()) if n else None))
                np.savez_compressed(folder/f'evaluation_scored_{arm}_t{t}.npz',costs_remaining=C,values=val,expected_degree=expected,exact_gradients=G)
            for left,right in [('sampled','before'),('hybrid','before'),('hybrid','sampled')]:
                difference=values[left,t]-values[right,t];mc=costs2[left,t]-costs2[right,t]
                for k,n in enumerate(core.names):
                    se=float(mc[:,[1,4].index(k)].std(ddof=1)/np.sqrt(len(mc))) if k in (1,4) else 0.
                    pairs.append(dict(dataset=ds,seed=seed,t=t,comparison=left+'-'+right,metric=n,difference=float(difference[k]),paired_MC_SE=se,analytic=k in (0,2,3)))
        for arm in protocol['arms']:
            rr=[x for x in results if x['dataset']==ds and x['seed']==seed and x['arm']==arm];base=[x for x in results if x['dataset']==ds and x['seed']==seed and x['arm']=='before']
            summary=dict(dataset=ds,seed=seed,arm=arm,updates=4)
            for key in [*core.names,'mean_five','CE','positive_CE','negative_CE','node_total_degree_MAE','node_relation_degree_MAE']:
                summary[key]=float(np.mean([x[key] for x in rr]));summary[key+'_change']=float(np.mean([x[key]-y[key] for x,y in zip(rr,base)]))
            summary['mean_gradient_norm']=float(np.mean([x['gradient_norm'] for x in updates if x['arm']==arm]));summary['mean_displacement_norm']=float(np.mean([x['displacement_norm'] for x in updates if x['arm']==arm]));summaries.append(summary)
        # Recompute estimator identities from the actual saved train samples.
        max_gradient_error=0.;max_value_error=0.
        for u in updates:
            z=np.load(folder/f'train_{u["arm"]}_u{u["update"]}.npz');p=z['probability'];bits=z['bits'];C=core.batch(bits);feedback=loo(p,bits,C);v=C.mean(0)
            if u['arm']=='hybrid':
                vv,gg,_=exact.value_gradient(p);v[[0,2,3]]=vv;feedback[[0,2,3]]=gg
            max_gradient_error=max(max_gradient_error,float(abs(feedback-z['feedback']).max()));max_value_error=max(max_value_error,float(abs(v-z['values']).max()))
            np.testing.assert_allclose(feedback,z['feedback'],atol=1e-12,rtol=1e-10);np.testing.assert_allclose(v,z['values'],atol=1e-12,rtol=1e-10)
        parent=torch.load(task['checkpoint'],map_location='cpu',weights_only=False);final=[];step_changes=[]
        for arm in protocol['arms']:
            state=torch.load(folder/f'{arm}_complete.pt',map_location='cpu',weights_only=False)['state'];assert state['updates']==state['ema']['updates']==2004
            for pid,s in state['optimizer']['state'].items():
                delta=float(s['step'])-float(parent['optimizer']['state'][pid]['step']);assert delta in (0.,4.);step_changes.append(delta)
                for key in ('exp_avg','exp_avg_sq'):assert torch.isfinite(s[key]).all()
            assert all(torch.isfinite(v).all() for v in state['model'].values() if torch.is_tensor(v));final.append(g.digest(state['rng']))
        assert final[0]==final[1]
        assert rt.sha(task['checkpoint'])==task['checkpoint_sha256'];assert rt.sha(task['data'])==task['data_sha256']
        checks.append(dict(dataset=ds,seed=seed,estimator_max_gradient_error=max_gradient_error,estimator_max_value_error=max_value_error,optimizer_step_increments=sorted(set(step_changes)),final_RNG_pairing=True,completed_and_EMA=2004,original_assets_unchanged=True))
    assert results,'no completed models'
    write('evaluation_summary.csv',results);write('relations.csv',relations);write('node_groups.csv',nodes);write('paired_cost_changes.csv',pairs);write('model_summary.csv',summaries)
    rt.save(OUT/'audit.json',dict(passed=True,models=checks,seconds=time.time()-start,model_calls=0,GPU_hours=0,scope='conditional clean probabilities; exact3 and K256 paired samples remaining2; no reverse chain; groups defined using known X0 reference',low_rule='within type reference degree<=2 excluding hub; hub>=within-type 90th percentile; other middle'))
    print(json.dumps(dict(summary=summaries,audit_seconds=time.time()-start),indent=2),flush=True)
if __name__=='__main__':main()
