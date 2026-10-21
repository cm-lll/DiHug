"""Freeze deterministic splits and clean supervised graphs BEFORE model checks."""
from collections import Counter
import itertools,time,resource
from types import SimpleNamespace
from common import *
sys.path.insert(0,str(EXP/'role_target_scalability_v1'))
from algorithms import flow_witness,gale_ryser
from role_condition.model import capacities,features as degree_features
from scipy.special import logsumexp
from capacity_readout import allocation_marginals,degree_logpmf

DEVELOPMENT={(8,6,4,3,2,1),(12,4,3,2,2,1),(10,10,1,1,1,1)}
ALLOWED=set(range(1,19))-{7,11}
HELDOUT={0,7,11}

def partitions(total=24,length=6,ceiling=18):
    def rec(rem,k,hi):
        if k==0:
            if rem==0:yield ()
            return
        for d in range(min(hi,rem),-1,-1):
            if rem-d>(k-1)*d:continue
            for rest in rec(rem-d,k-1,d):yield (d,)+rest
    return list(rec(total,length,ceiling))

def choose():
    allp=partitions();wide=[(4,4,4,4,4,4),(18,2,1,1,1,1)]
    available=[v for v in allp if set(v)<=ALLOWED and v not in DEVELOPMENT and v not in wide]
    # Predeclared greedy coverage, then max-min squared distance; ascending tuple breaks ties.
    while len(wide)<12:
        used=set(itertools.chain.from_iterable(wide))
        v=min(available,key=lambda v:(-len(set(v)-used),-min(sum((a-b)**2 for a,b in zip(v,w)) for w in wide),v))
        wide.append(v);available.remove(v)
    covered=set(itertools.chain.from_iterable(wide));assert covered==ALLOWED
    unseen=[v for v in allp if set(v)<=covered and v not in wide and v not in DEVELOPMENT]
    unseen.sort(key=lambda v:(sum(x*x for x in v),v))
    combo=[unseen[int((len(unseen)-1)*q/4)] for q in (1,2,3)]
    numeric=[]
    for value in (0,7,11):
        pool=[v for v in allp if value in v and set(v)<=(covered|{value}) and v not in DEVELOPMENT and v not in combo]
        pool.sort(key=lambda v:(sum(x*x for x in v),v));numeric.append(pool[len(pool)//2])
    new=sorted(wide[2:],key=lambda v:(sum(x*x for x in v),v));positive=new[(len(new)-1)//2]
    assert len(set(wide+combo+numeric))==18
    return wide,combo,numeric,positive

def freeze_panel():
    cfg=settings();wide,combo,numeric,positive=choose()
    oldrecords,oldtargets=base.load_panel(PANEL);base_target=oldtargets['balanced']
    names=['balanced','extreme']+[f'train_{i:02d}' for i in range(2,12)]
    vectors=dict(zip(names,wide));vectors.update({f'combo_{i+1}':v for i,v in enumerate(combo)})
    vectors.update({f'value_{value}':v for value,v in zip((0,7,11),numeric)})
    vectors.update(dev_single=(12,4,3,2,2,1),dev_double=(10,10,1,1,1,1))
    targets={}
    for name,vec in vectors.items():
        rows=list(base_target.roles);counts=Counter(vec)
        rows[3]=tuple((tuple(v if r==8 else 0 for r in range(10)),n) for v,n in sorted(counts.items()))
        targets[name]=base.RoleTarget(tuple(rows),base_target.type_counts,base_target.schema)
    positive_name=next(n for n in names if vectors[n]==positive)
    final=['balanced','extreme','dev_single','dev_double',positive_name]+[f'combo_{i}' for i in (1,2,3)]+['value_0','value_7','value_11']
    conditions=dict(rule='ALLOWED=1..18 excluding7/11; HELDOUT=0,7,11; sorted feasible partitions24/6/max18; greedy new-value coverage then max-min L2, ascending tuple tie; combo concentration quartiles; numeric concentration median; middle positive lower median concentration',
        allowed_training_values=sorted(ALLOWED),heldout_values=sorted(HELDOUT),
        vectors={n:list(v) for n,v in vectors.items()},targets={n:t.as_dict() for n,t in targets.items()},
        fingerprints={n:t.fingerprint for n,t in targets.items()},training=dict(narrow=names[:2],broad=names),
        combination_tests=[f'combo_{i}' for i in (1,2,3)],numerical_tests=['value_0','value_7','value_11'],
        numerical_interpretation={'value_0':'below positive training range: extrapolation','value_7':'within1..18 interpolation','value_11':'within1..18 interpolation'},
        positive_control=positive_name,final_targets=final,excluded_development=[list(v) for v in sorted(DEVELOPMENT)],frozen_before_any_new_learning=True)
    if (RESULT/'conditions.json').exists():assert read(RESULT/'conditions.json')==json.loads(json.dumps(conditions)),'frozen split cannot change'
    else:save(RESULT/'conditions.json',conditions)
    template=oldrecords[0]['case'];d=np.zeros((34,10),int)
    ff=template['space'].encode(template['clean'])[0]
    np.add.at(d,(template['clean'][0],ff),1);np.add.at(d,(template['clean'][1],ff),1)
    movies=np.flatnonzero(template['space'].types==2);genres=np.flatnonzero(template['space'].types==3)
    skeleton=set(map(tuple,template['clean'][:,ff!=8].T));graphs=[];evidence=[];selected={}
    def construct(vec,rng):
        mp=rng.permutation(18);gp=rng.permutation(6)
        ed,proof=flow_witness(d[movies[mp],8],np.asarray(vec)[gp])
        mg={(int(movies[mp[u]]),int(genres[gp[v]])) for u,v in ed}
        mg=base.swap(mg,rng,attempts=160)
        cc=base.graph_case(skeleton|mg)
        return cc,proof
    # All new graphs use same deterministic flow +160 attempted legal2switches factory.
    # Controls share first4 of the same24 new source topologies across arms.
    for i,name in enumerate(names):
        check();vec=vectors[name];rng=np.random.default_rng([cfg['graph_seed'],i]);train=[];val=[];attempts=0
        needed=24 if name in names[:2] else 4
        while len(train)<needed or len(val)<2:
            check();attempts+=1
            if attempts>cfg['graph_attempt_cap']:raise RuntimeError('bounded topology construction exhausted; no fallback')
            cc,proof=construct(vec,rng)
            assert base.extract_target(cc)==targets[name]
            assert np.array_equal(cc['counts'],template['counts'])
            if len(val)<2:
                if not any(base.isomorphic(cc,x) for x in val):val.append(cc)
                continue
            if any(base.isomorphic(cc,x) for x in val):continue
            # At least4 distinct clean sources in EVERY condition; remaining narrow
            # sources may repeat topologies and their frequencies are disclosed.
            # Extreme has at mostC(6,2)=15 unlabeled MG choices;24 distinct was impossible.
            if len(train)<4 and any(base.isomorphic(cc,x) for x in train):continue
            train.append(cc)
        selected[name]=(train,val)
        for arm in ('narrow','broad'):
            if arm=='narrow' and name not in names[:2]:continue
            pool=train if arm=='narrow' else train[:4]
            for j,cc in enumerate(pool):graphs.append(dict(id=f'{name}_source{j:02d}',arm=arm,condition=name,split='train',types=cc['space'].types.tolist(),edges=cc['clean'].T.tolist()))
        for j,cc in enumerate(val):graphs.append(dict(id=f'{name}_validation{j}',arm='shared',condition=name,split='validation',types=cc['space'].types.tolist(),edges=cc['clean'].T.tolist()))
        reps=[];frequencies=[]
        for cc in train:
            i=next((j for j,rep in enumerate(reps) if base.isomorphic(cc,rep)),None)
            if i is None:i=len(reps);reps.append(cc);frequencies.append(0)
            frequencies[i]+=1
        evidence.append(dict(condition=name,vector=list(vec),train_sources=len(train),distinct_train_topologies=len(reps),train_class_frequencies=frequencies,narrow_graphs=len(train) if name in names[:2] else 0,
             broad_graphs=4,validation_graphs=2,attempts=attempts,all_train_validation_nonisomorphic=True,
             complete_joint_fingerprint=targets[name].fingerprint,switch_attempts_per_construct=160,uniform_graph_sampling_claim=False))
    feasibility=[]
    for name,vec in vectors.items():
        ed,proof=flow_witness(d[movies,8],vec)
        witness=base.graph_case(skeleton|{(int(movies[u]),int(genres[v])) for u,v in ed})
        assert base.extract_target(witness)==targets[name]
        feasibility.append(dict(condition=name,**proof,complete_joint_verified=True,witness_used_in_inference=False))
    panel=dict(schema=template['space'].pairs,type_names=NAMES,type_counts=[4,6,18,6],graphs=graphs,
        factory='fixed original DMA skeleton; seeded row/column order integral bipartite flow;160 attempted degree-preserving legal swaps; type relabeling only after train/validation split',
        sampled_uniformly=False,train_graphs_per_arm=48,shared_control_training_topologies=8)
    save(RESULT/'panel.json',panel);save(RESULT/'construction.json',dict(graphs=evidence,feasibility=feasibility,all_feasible=True))
    save(RESULT/'resolved_config.json',cfg)
    save(RESULT/'split_binding.json',dict(conditions_sha256=sha(RESULT/'conditions.json'),panel_sha256=sha(RESULT/'panel.json'),
        construction_sha256=sha(RESULT/'construction.json'),time_frozen_epoch=time.time(),model_outputs_consulted=False))
    return oldrecords,targets

def correctness(oldrecords,targets):
    from scipy.stats import binom
    expected,environment=verify_inputs(SimpleNamespace(run=OLD,panel=PANEL),base)
    cfg=settings();values=base.fixed_validation(oldrecords,cfg);rows=[];allocations=[];encoding=[]
    case=oldrecords[0]['case'];inp=base.c.train_input(case,64,4,2026102124,0,np.random.default_rng(2026102124),force_t=8)
    for w in expected['weights']:
        check();m,_=make_model(case,cfg,w['seed']);m.load_state_dict(torch.load(w['path'],map_location='cpu',weights_only=True),strict=True)
        v=validate_device(m,values,cfg,{n:t for n,t in targets.items() if n in ('balanced','extreme')},check)
        old=read(OLD/f"seedjobs/seed{w['seed']}/seed{w['seed']}/normal2_validation_6000.json")
        maximum=0.
        assert len(v['rows'])==len(old['rows'])==12
        for a,b in zip(v['rows'],old['rows']):
            for k in ('id','condition','t','counts_by_family_label'):assert a[k]==b[k]
            for k in ('ce','ce_sum_by_family_label','expected_edges','expected_edge_bias'):
                np.testing.assert_allclose(a[k],b[k],atol=1e-6,rtol=1e-5)
                maximum=max(maximum,float(np.max(np.abs(np.asarray(a[k])-b[k]))))
        rows.append(dict(seed=w['seed'],weight_sha256=w['sha256'],maximum_validation_difference=maximum,old_mean=old['mean_ce'],new_mean=v['mean_ce']))
        # For all frozen targets, inspect actual nonzero trained encoder inputs and exact occupancy.
        m.eval()
        for name,targ in targets.items():
            check();m.set_target(targ);captures=[]
            hook=m.branch.role_encoder.register_forward_pre_hook(lambda mod,args:captures.append(args[0].detach().numpy().copy()))
            ctx=FrozenTypedContext(m,case['space'].types,inp['edges'],inp['edge_families'],'check')
            with torch.no_grad():_,info=m.branch(ctx,8/64,targ)
            hook.remove();ctx.close()
            cap=capacities(case['space'].types,targ.schema)
            for typ,role in enumerate(targ.roles):
                vec=np.array([v for v,n in role]);cnt=np.array([n for v,n in role])
                want=np.c_[degree_features(vec,cap[typ]),cnt/targ.type_counts[typ],np.log1p(cnt)/np.log1p(targ.type_counts[typ])].astype(np.float32)
                np.testing.assert_array_equal(captures[typ],want)
                att=info['allocation'][typ]['attention']
                np.testing.assert_allclose(att.sum(1),1,atol=2e-10,rtol=0)
                np.testing.assert_allclose(att.sum(0),cnt,atol=2e-9,rtol=0)
            allocations.append(dict(seed=w['seed'],target=name,encoder_exact=True,row_column_contract=True,max_DP_cells=max(x['cells'] for x in info['allocation'])))
            # Same-type permutation, same target; no network output requirement imposed.
            perm=np.arange(34);rng=np.random.default_rng(2026102125)
            for typ in range(4):
                ix=np.flatnonzero(case['space'].types==typ);perm[ix]=rng.permutation(ix)
            pc=FrozenTypedContext(m,case['space'].types,np.sort(perm[inp['edges']],axis=0),inp['edge_families'],'permuted')
            with torch.no_grad():_,pi=m.branch(pc,8/64,targ)
            pc.close()
            torch.testing.assert_close(pi['role_context'][torch.from_numpy(perm)],info['role_context'],atol=2e-6,rtol=1e-5)
            for typ in range(4):
                ix=np.flatnonzero(case['space'].types==typ);local=perm[ix]-ix[0]
                np.testing.assert_allclose(pi['allocation'][typ]['attention'][local],info['allocation'][typ]['attention'],atol=2e-10,rtol=0)
        del m
    # Independent finite-binomial convolution tests every legal clean Genre degree, no clipping/indexing.
    pmfs=[]
    for step in (0,1,8,32,64):
        a=base.c.schedule(64)[step];p=a+(1-a)*24/108;q=(1-a)*24/108
        for d in range(19):
            ref=np.convolve(binom.pmf(np.arange(d+1),d,p),binom.pmf(np.arange(19-d),18-d,q))
            actual=np.exp(degree_logpmf(18,d,a,24/108))
            np.testing.assert_allclose(actual,ref,atol=2e-12,rtol=0)
            pmfs.append(dict(t=step,degree=d,max_abs=float(abs(actual-ref).max())))
    save(RESULT/'correctness.json',dict(status='passed',validation=rows,condition_checks=allocations,Genre_likelihood=pmfs,
        computation_error_found=False,model_prediction_quality_not_a_gate=True,environment=environment,
        note='encoder vectors, unchanged capacity normalization and frequencies inspected; exact PMF and row/column sums; permutation checked using nonzero learned weights; no model change'))

if __name__=='__main__':
    start=time.monotonic();status='failed';reason=None
    try:
        assert not (RESULT/'panel.json').exists(),'do not replace frozen completed data'
        oldrecords,targets=freeze_panel();correctness(oldrecords,targets);check();status='completed'
    except BaseException as exc:reason=repr(exc);raise
    finally:save(RESULT/'preparation_receipt.json',dict(status=status,reason=reason,wall_seconds=time.monotonic()-start,
        elapsed_from_activation=time.time()-read(RESULT/'activation.json')['activation_epoch'],formal_updates=0,scientific_chains=0))
