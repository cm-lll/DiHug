"""Read-only old-chain replay and same-state comparison; no new sample law run."""
import argparse,contextlib,hashlib,json,os,resource,sys,time,traceback
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
HERE=Path(__file__).resolve().parent;EXP=HERE.parent
sys.path.insert(0,str(EXP/'role_condition_coverage_v1'))
import common as c
import capacity_readout as cap
from scaling import scale,required,ReadoutFailure
from device_eval import validate_device
from hsgd.computational import FrozenTypedContext

OUT=Path('/data2/lyh/gnn_project/DiHug/results/role_logscale_20261021_v1')
CFG=c.read(HERE/'config.json');ORIGINAL=cap.allocation_marginals
def check():
    if (OUT/'STOP').exists():raise InterruptedError('STOP')
    if time.time()>c.read(OUT/'activation.json')['deadline_epoch']-120:raise TimeoutError('unique60min reserve120')
def binding():
    f=c.read(c.RESULT/'freeze.json')
    for n,h in f['source'].items():assert c.sha(EXP/n)==h,n
    for n,h in f['inputs'].items():assert c.sha(c.RESULT/n)==h,n
    assert f['config']==c.settings()
    return f
def get_model(seed,case):
    f=binding();w=next(w for w in c.read(c.RESULT/'sealed_final_weights.json')['weights'] if w['arm']=='broad' and w['seed']==seed)
    assert c.sha(c.RESULT/w['path'])==w['sha256']
    ck=c.torch.load(c.RESULT/f'jobs/broad_{seed}/checkpoint_u006000.pt',map_location='cpu',weights_only=False)
    assert ck['binding']==f and ck['completed_updates']==6000
    m,_=c.make_model(case,c.settings(),seed)
    state=c.torch.load(c.RESULT/w['path'],map_location='cpu',weights_only=True)
    assert state.keys()==ck['model'].keys() and all(c.torch.equal(v,ck['model'][k]) for k,v in state.items())
    m.load_state_dict(state,strict=True);m.eval()
    env=c.base.c.environment('cpu');old=f['environment'] if 'environment' in f else c.read(c.RESULT/'correctness.json').get('environment',{})
    for k in ('python_version','packages','torch_runtime_version','float32_matmul_precision','matmul_allow_tf32','deterministic_algorithms','torch_threads'):
        assert env[k]==old[k],(k,env[k],old[k])
    return m,dict(weight=w,checkpoint_sha256=c.sha(c.RESULT/f'jobs/broad_{seed}/checkpoint_u006000.pt'),environment=env,source_verified=True)

def predict(m,case,state,target,t,seed,mode):
    records=[];captured=[];kernel=c.base.MarginalKernel(case['prior'],c.base.c.schedule(64))
    def solver(L,counts):
        tick=time.perf_counter()
        if mode=='exact':a,r=ORIGINAL(L,counts);r=dict(r,status='converged',seconds=time.perf_counter()-tick)
        else:a,r=required(L,counts,check)
        records.append(dict(ll=L.copy(),counts=counts.copy(),attention=a.copy(),receipt=r));return a,r
    def hook(module,args,value):captured.append(value[1])
    cap.allocation_marginals=solver;h=m.branch.register_forward_hook(hook)
    old,fam=c.base.c.state_edges(case,state);m.set_target(target);token=('paired',t,mode)
    p0=case['prior'].repeat(case['space'].counts).copy();coverage=c.np.zeros(len(state),int)
    plans=c.base.c.plan_for(case,[seed,t,61],4);planhash=hashlib.sha256();times={};tick=time.perf_counter()
    ctx=FrozenTypedContext(m,case['space'].types,old,fam,token)
    try:
        with c.torch.no_grad():
            for b in range(4):
                check();ids,f,q=plans.block(b)
                for ar in (ids,f,q):planhash.update(c.np.ascontiguousarray(ar).tobytes())
                _,p=c.base.predict(m,case['space'].types,old,fam,q,f,t/64,context=ctx,token=token,times=times,details=True)
                p0[ids]=c.base.c.check_prob(p,f,case);coverage[ids]+=1
    finally:ctx.close();h.remove();cap.allocation_marginals=ORIGINAL
    assert len(records)==4 and len(captured)==1
    assert c.np.array_equal(coverage,case['active'].repeat(case['space'].counts))
    families=c.np.repeat(c.np.arange(10),case['space'].counts)
    q=kernel.reverse(p0,state,families,t-1,t)
    diag=captured[0];context=diag['role_context'].detach().cpu().numpy()
    return dict(p0=p0,q=q,records=records,context=context,query_sha256=planhash.hexdigest(),
        seconds=time.perf_counter()-tick,times=times,
        branch_seconds={k:float(v) for k,v in diag.items() if k.endswith('_seconds')})

def compare(seed,name,t,state,m,case,target,out,expected):
    a=predict(m,case,state,target,t,2026102031,'exact')
    assert a['query_sha256']==expected['query_sha256']
    try:b=predict(m,case,state,target,t,2026102031,'scaled')
    except ReadoutFailure as exc:
        c.save(out/f'{name}_t{t}.json',dict(status='not_converged',reason=str(exc),condition=name,t=t,seed=seed));return
    assert a['query_sha256']==b['query_sha256']
    arrays=dict(state=state,p0_exact=a['p0'],p0_scaled=b['p0'],q_exact=a['q'],q_scaled=b['q'],
        context_exact=a['context'],context_scaled=b['context'])
    allocation=[]
    for typ,(x,y) in enumerate(zip(a['records'],b['records'])):
        assert c.np.array_equal(x['ll'],y['ll']) and c.np.array_equal(x['counts'],y['counts'])
        p,r=x['attention'],y['attention'];vec=c.np.array([v for v,n in target.roles[typ]])
        def entropy(z):
            v=z[z>0];return float(-(v*c.np.log(v)).sum()/len(z))
        item=dict(type=typ,n=len(p),k=p.shape[1],row_TV_mean=float(abs(p-r).sum(1).mean()/2),
          row_TV_max=float(abs(p-r).sum(1).max()/2),max_abs=float(abs(p-r).max()),
          entropy_exact=entropy(p),entropy_scaled=entropy(r),
          soft_target_degree_max_abs=float(abs((p-r)@vec).max()),
          role_context_max_abs=float(abs(a['context'][case['space'].types==typ]-b['context'][case['space'].types==typ]).max()),
          exact=x['receipt'],scaled=y['receipt'])
        # Warm repeated solver calls on identical LL; no network and no RNG.
        warm={}
        for label,fn in [('exact',ORIGINAL),('scaled',required)]:
            ticks=[]
            for j in range(3):
                check();tick=time.perf_counter();fn(x['ll'],x['counts']);ticks.append(time.perf_counter()-tick)
            warm[label]=ticks
        item['repeated_solver_seconds']=warm;allocation.append(item)
        for key,val in [('LL',x['ll']),('counts',x['counts']),('exact',p),('scaled',r)]:arrays[f'type{typ}_{key}']=val
    families=c.np.repeat(c.np.arange(10),case['space'].counts);rels=[]
    for rel in range(10):
        ix=families==rel;before=state[ix];qa=a['q'][ix];qb=b['q'][ix]
        rels.append(dict(family=rel,n=int(ix.sum()),p0_max_abs=float(abs(a['p0'][ix]-b['p0'][ix]).max(initial=0)),
            p0_mean_abs=float(abs(a['p0'][ix]-b['p0'][ix]).mean()) if ix.any() else None,
            q_max_abs=float(abs(qa-qb).max(initial=0)),expected_exact=float(qa.sum()),expected_scaled=float(qb.sum()),
            births_exact=float(qa[before==0].sum()),births_scaled=float(qb[before==0].sum()),
            deaths_exact=float((1-qa[before==1]).sum()),deaths_scaled=float((1-qb[before==1]).sum()),
            net_exact=float(qa.sum()-before.sum()),net_scaled=float(qb.sum()-before.sum())))
    c.np.savez_compressed(out/f'{name}_t{t}.npz',**arrays)
    c.save(out/f'{name}_t{t}.json',dict(status='completed',seed=seed,condition=name,t=t,state_sha256=c.base.sha(state.tolist()),
        query_sha256=a['query_sha256'],allocation=allocation,relations=rels,
        exact_seconds=a['seconds'],scaled_seconds=b['seconds'],exact_timing=a['times'],scaled_timing=b['times'],
        exact_branch=a['branch_seconds'],scaled_branch=b['branch_seconds'],independent_samples_added=0))

def run(seed):
    start=time.perf_counter();out=OUT/f'seed{seed}';out.mkdir();status='failed';reason=None
    try:
        check();records,targets=c.load_data();case=c.metadata_case(c.base,c.PANEL,targets);m,bind=get_model(seed,case);c.save(out/'binding.json',bind)
        # Full original fixed validation, same actual inputs/targets/normalization.
        v=validate_device(m,c.base.fixed_validation(records,c.settings()),c.settings(),targets,check)
        old=c.read(c.RESULT/f'jobs/broad_{seed}/validation_u006000.json')
        assert len(v['rows'])==len(old['rows'])
        maximum=0.
        for x,y in zip(v['rows'],old['rows']):
            assert (x['id'],x['condition'],x['t'])==(y['id'],y['condition'],y['t'])
            for k in ('ce','ce_sum_by_family_label','expected_edges'):
                maximum=max(maximum,float(abs(c.np.asarray(x[k])-c.np.asarray(y[k])).max()))
        assert maximum<=2e-8,maximum
        c.save(out/'validation.json',dict(maximum_difference=maximum,tolerance=2e-8,validation=v))
        for name in c.read(c.RESULT/'conditions.json')['final_targets']:
            check();stem=f'{name}_2026102031';source=c.RESULT/f'jobs/broad_{seed}/generated'
            oldsteps=c.read(source/(stem+'_steps.json'));result=c.read(source/(stem+'_result.json'))
            assert c.sha(source/(stem+'.npz'))==result['graph_sha256'] and c.sha(source/(stem+'_steps.json'))==result['steps_sha256']
            saved={64:(c.np.random.default_rng(2026102031).random(case['space'].total)<case['prior'].repeat(case['space'].counts)).astype(c.np.uint8)}
            assert c.base.sha(saved[64].tolist())==result['initial_state_sha256']
            def progress(state,steps,rng):
                x=steps[-1];y=oldsteps[len(steps)-1]
                for k in ('t','uniform_sha256','query_sha256','rng_after','state_sha256','edge_count','exact_coverage','covered_slots'):assert x[k]==y[k],(name,k,x[k],y[k])
                if x['t']-1 in CFG['diagnostic_times']:saved[x['t']-1]=state.copy()
            state,steps,initial=c.original_chain(m,case,targets[name],2026102031,c.base,check,progress)
            assert steps[-1]['state_sha256']==oldsteps[-1]['state_sha256'] and set(saved)==set(CFG['diagnostic_times'])
            for t in CFG['diagnostic_times']:
                compare(seed,name,t,saved[t],m,case,targets[name],out,next(x for x in oldsteps if x['t']==t))
            c.save(out/(name+'_replay.json'),dict(all64_hashes_match=True,initial_hash=initial,source_graph_sha256=result['graph_sha256'],independent_new_samples=0))
            print(seed,name,'replayed and compared',flush=True)
        status='completed'
    except BaseException as exc:reason=repr(exc);traceback.print_exc()
    finally:c.save(out/'receipt.json',dict(status=status,reason=reason,seconds=time.perf_counter()-start,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024))
    return int(status!='completed')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);sys.exit(run(p.parse_args().seed))
