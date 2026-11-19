"""Reuse scored baseline; evaluate only new common-direction endpoint."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from worker import *
import csv
def read(p):return list(csv.DictReader(Path(p).open()))
def write(n,rows):
    with (OUT/n).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    start=time.time();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    protocol=json.loads((OUT/'protocol.json').read_text());rows=[];relations=[];nodes=[];pairs=[];mechanism=[];checks=[]
    for row in protocol['tasks']:
        ds,seed=row['dataset'],row['seed'];folder=OUT/f'{ds}_{seed}';old=OLD/folder.name
        if not (folder/'checks.json').exists():continue
        fixed=torch.load(old/'evaluation_inputs.pt',map_location='cpu',weights_only=False)['inputs'];data=np.load(row['data']);inp=fixed[0];ix=np.argsort(inp['ids']);core=HardScore(data['types'],inp['query'][:,ix],inp['f'][ix],inp['labels'][ix],len(data['schema']));exact=Exact(core)
        ev=json.loads((folder/'evaluation.json').read_text())
        for j,t in enumerate(protocol['evaluation_times']):
            pred=np.load(folder/f'eval_common_t{t}.npz');p=pred['probability'];V,_,stats=exact.value_gradient(p)
            uniform=np.random.default_rng(seed+protocol['evaluation_hard_seed_offset']+j).random((256,len(p)));bits=(uniform<p).astype(np.uint8);C=remaining(core,bits);val=np.zeros(5);val[[0,2,3]]=V;val[[1,4]]=C.mean(0)
            info=next(x for x in ev if x['t']==t);assert info['input_hash']==rt.input_hash([fixed[j]])
            expected=np.zeros((core.n,core.r));np.add.at(expected,(core.q[0],core.f),p);np.add.at(expected,(core.q[1],core.f),p);ref=core.ref['degree']
            rows.append(dict(dataset=ds,seed=seed,arm='common',t=t,**{n:float(v) for n,v in zip(core.names,val)},mean_five=float(val.mean()),CE=info['CE'],positive_CE=info['positive_CE'],negative_CE=info['negative_CE'],expected_edges=float(p.sum()),expected_triangles=sum(x['mean'] for x in stats['triangles']),node_total_degree_MAE=float(abs(expected.sum(1)-ref.sum(1)).mean()),node_relation_degree_MAE=float(abs(expected-ref).mean())))
            for f in range(core.r):
                ix=core.f==f;pos=ix&(core.reference==1);neg=ix&(core.reference==0);ii=next(x for x in info['relations'] if x['family']==f)
                relations.append(dict(dataset=ds,seed=seed,arm='common',t=t,family=f,support=int(ix.sum()),reference_edges=int(core.ref['counts'][f]),expected_edges=float(p[ix].sum()),missed_positive_mass=float((1-p[pos]).sum()),false_positive_mass=float(p[neg].sum()),positive_CE=ii['positive_CE'],negative_CE=ii['negative_CE']))
            for ty in np.unique(core.types):
                mask=core.types==ty;d=ref.sum(1);cut=float(np.quantile(d[mask],.9));hub=mask&(d>=cut);low=mask&(d<=2)&~hub;mid=mask&~hub&~low
                for label,selected in [('low',low),('middle',mid),('hub',hub)]:
                    n=int(selected.sum());nodes.append(dict(dataset=ds,seed=seed,arm='common',t=t,node_type=int(ty),group=label,n=n,reference_hub_threshold=cut,total_degree_MAE=float(abs(expected.sum(1)-d)[selected].mean()) if n else None,relation_degree_MAE=float(abs(expected-ref)[selected].mean()) if n else None,mean_expected_degree=float(expected.sum(1)[selected].mean()) if n else None,mean_reference_degree=float(d[selected].mean()) if n else None))
            np.savez_compressed(folder/f'evaluation_scored_common_t{t}.npz',values=val,costs_remaining=C,expected_degree=expected)
            for baseline in ('before','hybrid'):
                z=np.load(old/f'evaluation_scored_{baseline}_t{t}.npz');diff=val-z['values'];cd=C-z['costs_remaining']
                for k,n in enumerate(core.names):pairs.append(dict(dataset=ds,seed=seed,t=t,comparison='common-'+baseline,metric=n,difference=float(diff[k]),paired_MC_SE=float(cd[:,[1,4].index(k)].std(ddof=1)/16) if k in (1,4) else 0.,analytic=k in (0,2,3)))
        updates=json.loads((folder/'updates.json').read_text());maxg=0.;maxkkt=0.;aux=np.random.default_rng(seed+protocol['training_hard_seed_offset'])
        for u in updates:
            z=np.load(folder/f'train_common_u{u["update"]}.npz');p=z['probability'];vals,grads,_=exact.value_gradient(p);cost=remaining(core,z['bits']);fb=np.zeros_like(z['feedback']);fb[[0,2,3]]=grads;fb[[1,4]]=loo(p,z['bits'],cost);maxg=max(maxg,float(abs(fb-z['feedback']).max()));np.testing.assert_allclose(fb,z['feedback'],rtol=1e-10,atol=1e-12)
            coeff,cert=solver(z['Gram']);np.testing.assert_allclose(coeff,z['coefficients'],rtol=1e-9,atol=1e-9);maxkkt=max(maxkkt,cert['residual'])
            np.testing.assert_allclose(z['Gram']@coeff,z['combined_inner_products'],rtol=1e-4,atol=1e-8)
            assert min(u['combined_cosines'])>=-1e-5
            np.testing.assert_allclose(sum(np.asarray(x['actual_directional_derivatives']) for x in u['modules'].values()),u['actual_directional_derivatives'],rtol=1e-12,atol=1e-12)
            draw=aux.random((64,len(p)));assert g.digest(draw)==u['draw_hash'];assert g.digest(aux.bit_generator.state)==u['aux_after_hash']
            for k,n in enumerate(core.names):mechanism.append(dict(dataset=ds,seed=seed,update=u['update'],t=u['t'],metric=n,gradient_norm=u['group_gradient_norms'][k],coefficient=u['coefficients'][k],combined_gradient_inner=u['combined_inner_products'][k],combined_gradient_cosine=u['combined_cosines'][k],actual_update_inner=u['actual_directional_derivatives'][k],actual_update_cosine=u['actual_update_cosines'][k],actual_first_order_harm=u['actual_update_cosines'][k]>1e-5,correction_relative_norm=u['correction_relative_norm'],solver_residual=u['solver']['residual'],combined_gradient_norm=u['gradient_norm'],displacement_norm=u['displacement_norm']))
        final=torch.load(folder/'common_complete.pt',map_location='cpu',weights_only=False);refstate=torch.load(old/'hybrid_complete.pt',map_location='cpu',weights_only=False);parent=torch.load(row['checkpoint'],map_location='cpu',weights_only=False)
        assert final['state']['updates']==final['state']['ema']['updates']==2004;assert g.digest(final['state']['rng'])==g.digest(refstate['state']['rng']);assert g.digest(final['auxiliary_rng'])==g.digest(refstate['auxiliary_rng'])==g.digest(aux.bit_generator.state)
        increments=[]
        for pid,s in final['state']['optimizer']['state'].items():
            d=float(s['step'])-float(parent['optimizer']['state'][pid]['step']);assert d in (0,4);increments.append(d)
            for key in ('exp_avg','exp_avg_sq'):assert torch.isfinite(s[key]).all()
        checks.append(dict(dataset=ds,seed=seed,max_feedback_error=maxg,max_solver_residual=maxkkt,optimizer_increments=sorted(set(increments)),full_rng_pairing=True,aux_draws_equal=True))
    for path,h in json.loads((OUT/'bindings.json').read_text()).items():assert rt.sha(path)==h,path
    for name,new in [('evaluation_summary.csv',rows),('relations.csv',relations),('node_groups.csv',nodes)]:
        oldrows=[r for r in read(OLD/name) if r['arm'] in ('before','hybrid')];write(name,oldrows+new)
    write('paired_cost_changes.csv',pairs);write('mechanism.csv',mechanism)
    summaries=[]
    for task in protocol['tasks']:
        ds,seed=task['dataset'],task['seed']
        for arm in ('before','hybrid','common'):
            rr=[x for x in read(OUT/'evaluation_summary.csv') if x['dataset']==ds and int(x['seed'])==seed and x['arm']==arm]
            if not rr:continue
            summaries.append(dict(dataset=ds,seed=seed,arm=arm,**{k:float(np.mean([float(x[k]) for x in rr])) for k in [*HardScore.names,'mean_five','CE','positive_CE','negative_CE','expected_edges','expected_triangles','node_total_degree_MAE','node_relation_degree_MAE']}))
    write('model_summary.csv',summaries);rt.save(OUT/'audit.json',dict(passed=True,models=checks,seconds=time.time()-start,all_bound_files_unchanged=True,model_calls=0,GPU_hours=0));print(json.dumps(summaries,indent=2))
if __name__=='__main__':main()
