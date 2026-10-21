import resource,sys,time
from large_common import *
def prepare():
    start=time.perf_counter();out=[];compat=[];weights=read(c.RESULT/'sealed_final_weights.json')['weights'];small=next(w for w in weights if w['arm']=='broad' and w['seed']==2026102011)
    records,targets=c.load_data();smallcase=c.metadata_case(c.base,c.PANEL,targets);bindings=[]
    for seed in g.SEEDS:
        check();m,b=g.old.get_model(seed,smallcase);bindings.append(b);del m
    save(OUT/'existing_weights_source_binding.json',bindings)
    wp=c.RESULT/small['path'];assert sha(wp)==small['sha256'];state=torch.load(wp,map_location='cpu',weights_only=True)
    for index,name in enumerate(('IMDB','PubMed')):
        check();case,target,cfg,inv,path=dataset(name);model=make_model(case,cfg);random_hash=state_digest(model)
        initial_prior=model.prior.clone();status='strict_tensor_load_passed';error=None
        try:model.load_state_dict(state,strict=True)
        except RuntimeError as exc:status='strict_tensor_load_failed';error=str(exc)
        semantic=status=='strict_tensor_load_passed' and torch.equal(initial_prior,model.prior) and c.settings()['T']==1000
        assert not semantic,'unexpected compatible trained configuration; stop before choosing probe weights'
        compat.append(dict(dataset=name,source_weight=small,source_weight_hash_checked=True,tensor_load=status,error=error,
            source_T=c.settings()['T'],target_T=1000,prior_equal=status=='strict_tensor_load_passed' and torch.equal(initial_prior,model.prior),
            scientifically_compatible=False,action='fresh fixed-seed initialization, no transferred parameters, no quality evaluation'))
        model=make_model(case,cfg);assert state_digest(model)==random_hash
        dims=dict(c.base.c.dimensions(model),degree_encoder_input=model.branch.degree_encoder[0].in_features,
            role_encoder_input=model.branch.role_encoder[0].in_features,role_fusion_input=model.role_fusion.in_features,
            joint_head_input=model.joint_head[0].in_features,parameter_count=sum(p.numel() for p in model.parameters()))
        params=sum(p.numel()*p.element_size() for p in model.parameters());capm=cap.capacities(case['space'].types,case['space'].pairs)
        save(OUT/f'{name}_binding.json',dict(dataset=name,data_sha256=sha(path),source_config_path=str(path.parent.parent/'configs'/f'{name}.json'),source_config_sha256=sha(path.parent.parent/'configs'/f'{name}.json'),
            config=cfg,model_architecture=model.architecture_version,random_seed=SEED,random_weight_state_sha256=random_hash,dimensions=dims,
            random_weights_cannot_evaluate_quality=True,anonymous_target=target.as_dict(),target_sha256=target.fingerprint,environment=c.base.c.environment('cpu')))
        for t in (20,500,1000):
            check();tick=time.perf_counter();ei,ef,_=c.base.c.global_noisy_edges(case['space'],case['clean'],case['prior'],c.base.c.schedule(1000)[t],np.random.default_rng([NOISE_SEED,index,t]))
            f,loc=case['space'].encode(ei);st=np.zeros(case['space'].total,np.uint8);st[case['space'].offsets[f]+loc]=1
            degree=np.bincount(ei.ravel(),minlength=len(case['space'].types));wedges=int(np.sum(degree*(degree-1)//2))
            plan=c.base.c.plan_for(case,[DRAW_SEED+index,t,61],20);inventory=c.base.c.query_inventory(case,plan)
            sizes=inventory['query_sizes'];planhash=hashlib.sha256()
            for b in range(20):
                for x in plan.block(b):planhash.update(np.ascontiguousarray(x).tobytes())
            pathin=OUT/f'{name}_t{t}_input.npz';np.savez_compressed(pathin,state=st,edges=ei,families=ef,types=case['space'].types,schema=np.asarray(case['space'].pairs))
            matrices=[dict(type=typ,n=target.type_counts[typ],k=len(rows),elements=target.type_counts[typ]*len(rows),workspace_bytes=target.type_counts[typ]*len(rows)*128+len(rows)**2*32) for typ,rows in enumerate(target.roles)]
            directed=2*(len(ef)+max(sizes));network=directed*256*4*16+len(st)*64+min(wedges,2000000)*128+len(case['space'].types)*256*4*16
            estimate=512*1024**2+params+network+max(x['workspace_bytes'] for x in matrices)+inv['likelihood_cache_bytes_per_time']
            gates=dict(wedges=wedges<=cfg['model']['max_wedges'],Newton_memory=all(x['elements']<=2000000 and x['workspace_bytes']<=256*1024**2 for x in matrices),estimated_RSS=estimate<=6*1024**3)
            row=dict(dataset=name,t=t,path=pathin.name,sha256=sha(pathin),noise_seed=[NOISE_SEED,index,t],draw_seed=DRAW_SEED+index,
                forward_noise_only=True,not_free_generation_state=True,noise_seconds=time.perf_counter()-tick,physical_noise_edges=len(ef),raw_wedges=wedges,
                query=inventory,query_sha256=planhash.hexdigest(),matrices=matrices,estimated_parameters_bytes=params,
                estimated_state_and_trace_bytes=len(st)*64,estimated_directed_computational_edges=directed,estimated_network_and_CN_workspace_bytes=network,
                estimated_likelihood_cache_bytes=inv['likelihood_cache_bytes_per_time'],estimated_RSS_upper_engineering_bytes=estimate,resource_gates=gates,resource_gate_pass=all(gates.values()))
            out.append(row);print(name,t,'edges',len(ef),'wedge',wedges,'query',min(sizes),max(sizes),'estMiB',round(estimate/1024**2),gates,flush=True)
        del model
    save(OUT/'weight_compatibility.json',compat);save(OUT/'input_manifest.json',out)
    save(OUT/'preparation_receipt.json',dict(status='passed',inputs=6,seconds=time.perf_counter()-start,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,trained_weights_used=False,random_weights_only=True))
if __name__=='__main__':prepare()
