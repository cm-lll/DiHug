"""Bounded VJP consistency of cached K64 feedback. No optimizer steps or sampling."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,time,json,csv,signal,traceback,importlib.util,gc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/real_feedback_parameter_check_20261110';OLD=ROOT/'results/hard_clean_pilot_v2_20261110';CPU=ROOT/'results/real_hard_feedback_cost_20261110'
OUT=Path(os.environ.get('DIHUG_DIAG_OUT',str(OUT)))
spec=importlib.util.spec_from_file_location('original_runtime',ROOT/'research_branches/hard_clean_pilot_v2_20261110/runtime.py');rt=importlib.util.module_from_spec(spec);spec.loader.exec_module(rt)
torch,np,g=rt.torch,rt.np,rt.g
def dump(name,z):rt.save(OUT/name,z)
def group(name):
    if name.startswith('backbone.'):return 'shared_backbone'
    if name.startswith('common_neighbor.'):return 'CN_exclusive'
    if name.startswith('joint_head.'):return 'joint_head'
    if name.startswith('branch.degree'):return 'degree_branch'
    return 'other'
def main():
    start=time.time();OUT.mkdir(exist_ok=False);rt.setup();rows=[];bindings={};done=[]
    receipt=dict(start_epoch=start,status='running',forwards=0,VJPs=0,updates=0,new_hard_graphs=0,reverse_chains=0)
    def check():
        if time.time()-start>660:raise TimeoutError('11 minute science line; reserve GPU release')
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('12 minute GPU process cap')));signal.alarm(710)
    try:
        assert torch.cuda.is_available() and torch.cuda.device_count()==1
        assert '2080 Ti' in torch.cuda.get_device_name(0)
        env=g.common.environment('cuda');dump('environment.json',env)
        protocol=dict(datasets={'IMDB':[2026110411,2026110412],'PubMed':[2026110421,2026110422]},times=[1,250,800],updates=2000,readout='raw eval',cached_K=64,cached_repeats=4,
            max_forwards=16,max_VJPs=268,parameter_updates=0,new_samples=0,
            method='VJP J^T g from frozen cached margin gradients; 20 per condition plus analytic count and first-repeat five-group mean. Fresh-forward mean replay at t1 per model.',
            group_rule='CN exclusive=common_neighbor.*, shared first-layer context counted once in backbone; joint_head.*, branch.degree*, remaining other',
            probability_tolerance=2e-6,probability_tolerance_scope='cache-vs-restored forward identity only, NOT previous update replay acceptance or bitwise trajectory claim',
            numerical_linearity_relative_tolerance=1e-3,selection='all 4 existing regular models, predeclared low/mid/high times; no quality selection',
            budgets=dict(wall_minutes=15,GPU_hours=.20,GPU_count=1,science_seconds=660,process_kill_seconds=720,CPU_threads=1,RSS_GiB=4),
            limits=['parameter gradient not AdamW displacement','same forward-corrupted reference inputs, not free generation','four Monte Carlo repeats are not training replicates','no inference of quality from direction agreement','old probability draws reused; numerical cache mismatch recorded, no fresh on-policy claim'])
        dump('protocol.json',protocol);dump('freeze.json',dict(epoch=time.time(),sha256=rt.sha(OUT/'protocol.json')))
        old=json.loads((OLD/'bindings.json').read_text())
        for path,h in old.items():
            if path.endswith('.py'):assert rt.sha(path)==h,'source changed '+path;bindings[path]=h
        for path in [Path(__file__),Path(__file__).with_name('watch.py'),CPU/'protocol.json',CPU/'audit.json',CPU/'assets.json']:
            bindings[str(path)]=rt.sha(path)
        assert json.loads((CPU/'audit.json').read_text())['passed']
        for ds,seeds in protocol['datasets'].items():
            for seed in seeds:
                if os.environ.get('DIHUG_MODEL_SEED') and seed!=int(os.environ['DIHUG_MODEL_SEED']):continue
                check();folder=OLD/f'{ds}_{seed}_HARD_MGDA';identity=json.loads((folder/'identity.json').read_text());row=identity['row']
                z,cfg,case,m,o,e,rng,b=rt.load(row);m.eval();params=list(m.parameters());names=[n for n,_ in m.named_parameters()]
                for path in [Path(row['checkpoint']),Path(row['data']),folder/'identity.json',folder/'evaluation_inputs.pt']:bindings[str(path)]=rt.sha(path)
                fixed=torch.load(folder/'evaluation_inputs.pt',map_location='cpu',weights_only=False)
                baseline=g.digest(dict(model=m.state_dict(),optimizer=o.state_dict(),ema=e.__dict__,rng=g.rng_capture(rng,'cuda'),completed=b.completed))
                layout=[];offset=0
                for n,p in zip(names,params):layout.append(dict(name=n,group=group(n),start=offset,end=offset+p.numel(),shape=list(p.shape)));offset+=p.numel()
                dump(f'{ds}_{seed}_layout.json',layout)
                for t in protocol['times']:
                    check();tick=time.time();tag=f'{ds}_{seed}_t{t}';inp=next(x for x in fixed['inputs'] if x['t']==t);ih=rt.input_hash([inp]);assert ih==fixed['hashes'][[x['t'] for x in fixed['inputs']].index(t)]
                    cp=CPU/f'{tag}.npz';bindings[str(cp)]=rt.sha(cp);saved=np.load(cp);order=np.argsort(inp['ids']);np.testing.assert_array_equal(saved['ids'],inp['ids'][order])
                    before_rng=g.rng_capture(rng,'cuda');torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();tf=time.perf_counter();token=('VJP',seed,t)
                    ctx=g.FrozenTypedContext(m,case['space'].types,inp['edges'],inp['edge_families'],token)
                    graph,logits,p=ctx.predict(inp['query'],inp['f'],t/1000,token,details=True);receipt['forwards']+=1
                    margin=(logits[:,1]-logits[:,0])[torch.as_tensor(order,device='cuda')];prob=p.detach().cpu().double().numpy()[order];cached=saved['probability'];gap=float(abs(prob-cached).max())
                    assert gap<=protocol['probability_tolerance'],'cache prediction mismatch; stop, no resampling'
                    assert np.isfinite(prob).all() and np.isfinite(saved['logit_gradients']).all()
                    sigmoid_error=float(abs(torch.sigmoid(margin.detach().double()).cpu().numpy()-prob).max());assert sigmoid_error<=2e-6
                    torch.cuda.synchronize();forward_seconds=time.perf_counter()-tf
                    labels=inp['labels'][order];families=inp['f'][order];analytic=np.zeros_like(prob)
                    for f in range(case['n_families']):
                        ix=families==f;count=int(labels[ix].sum());mu=float(prob[ix].sum());analytic[ix]=prob[ix]*(1-prob[ix])*(1-2*prob[ix]+2*(mu-count))/(max(1,count)**2*case['n_families'])
                    cot=np.concatenate([saved['logit_gradients'].reshape(20,-1),analytic[None],saved['logit_gradients'][0].mean(0)[None]],axis=0)
                    vectors=[];uses=None;back_seconds=[]
                    def vjp(marg,v,keep):
                        nonlocal uses
                        torch.cuda.synchronize();ts=time.perf_counter();gs=torch.autograd.grad(marg,params,grad_outputs=torch.as_tensor(v,device='cuda',dtype=marg.dtype),allow_unused=True,retain_graph=keep);receipt['VJPs']+=1
                        this=[x is not None for x in gs]
                        if uses is None:uses=this
                        assert uses==this,'parameter dependency support changed'
                        ans=torch.cat([(torch.zeros_like(pp) if gg is None else gg).detach().reshape(-1) for pp,gg in zip(params,gs)]);assert torch.isfinite(ans).all();torch.cuda.synchronize();back_seconds.append(time.perf_counter()-ts);return ans
                    for j,v in enumerate(cot):check();vectors.append(vjp(margin,v,j<21))
                    ctx.close();del ctx,graph,logits,p,margin
                    linear_rel=float((vectors[21]-torch.stack(vectors[:5]).mean(0)).double().norm()/vectors[21].double().norm().clamp_min(1e-30));assert linear_rel<1e-3,'VJP linearity failure'
                    replay_rel=None;replay_probability_gap=None
                    if t==1:
                        g.rng_restore(before_rng,rng,'cuda');tok=('fresh_replay',seed,t);cx=g.FrozenTypedContext(m,case['space'].types,inp['edges'],inp['edge_families'],tok)
                        gr,ll,pr=cx.predict(inp['query'],inp['f'],t/1000,tok,details=True);receipt['forwards']+=1;mm=(ll[:,1]-ll[:,0])[torch.as_tensor(order,device='cuda')]
                        replay_probability_gap=float(abs(pr.detach().cpu().double().numpy()[order]-prob).max());vv=vjp(mm,cot[21],False);vectors.append(vv)
                        replay_rel=float((vv-vectors[21]).double().norm()/vectors[21].double().norm().clamp_min(1e-30));cx.close();del cx,gr,ll,pr,mm,vv
                    V=torch.stack(vectors);del vectors
                    # Float64 Gram accumulation in bounded chunks, partitioning shared parameters once.
                    grams={k:np.zeros((len(V),len(V))) for k in sorted(set(x['group'] for x in layout))}
                    for item in layout:
                        for st in range(item['start'],item['end'],100000):
                            W=V[:,st:min(st+100000,item['end'])].double();grams[item['group']]+=(W@W.T).cpu().numpy();del W
                    total=sum(grams.values());margin_gram=cot@cot.T
                    np.savez_compressed(OUT/f'{tag}_grams.npz',parameter=total,margin=margin_gram,**grams)
                    del V;torch.cuda.synchronize()
                    assert all(pp.grad is None for pp in params),'autograd.grad polluted training gradients'
                    assert g.digest(g.rng_capture(rng,'cuda'))==g.digest(before_rng),'diagnostic consumed RNG'
                    after=g.digest(dict(model=m.state_dict(),optimizer=o.state_dict(),ema=e.__dict__,rng=g.rng_capture(rng,'cuda'),completed=b.completed));assert baseline==after,'model/Adam/EMA/RNG state mutated'
                    info=dict(dataset=ds,seed=seed,t=t,source_input_hash=ih,checkpoint=row['checkpoint'],checkpoint_sha256=row['checkpoint_sha256'],parameter_count=offset,cache_probability_max_difference=gap,sigmoid_contract_max_difference=sigmoid_error,forward_seconds=forward_seconds,VJP_seconds_mean=float(np.mean(back_seconds)),VJPs=len(back_seconds),condition_seconds=time.time()-tick,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,linearity_relative_error=linear_rel,fresh_forward_VJP_relative_difference=replay_rel,fresh_forward_probability_difference=replay_probability_gap,complete_state_unchanged=True)
                    rows.append(info);done.append(tag);dump('conditions.json',rows);dump('receipt.json',dict(receipt,completed_conditions=done,elapsed_seconds=time.time()-start));print(json.dumps(info),flush=True)
                    if len(rows)==1:
                        estimate=(time.time()-start)+11*info['condition_seconds']*1.5+45;dump('calibration.json',dict(first=info,remaining_conditions=11,safety_factor=1.5,estimated_total_seconds=estimate,limit_seconds=660))
                        if estimate>660:raise TimeoutError('full fixed panel cannot fit measured budget; stop without reducing seeds')
                    del saved,cot,total,grams;gc.collect();torch.cuda.empty_cache()
                del z,case,m,o,e,rng,b,params;gc.collect();torch.cuda.empty_cache()
        for path,h in bindings.items():assert rt.sha(path)==h,'bound source/asset changed '+path
        dump('bindings.json',bindings);receipt['status']='completed'
    except BaseException as exc:
        receipt['status']='failed';receipt['error']=repr(exc);(OUT/'failure.txt').write_text(traceback.format_exc());raise
    finally:
        receipt.update(elapsed_seconds=time.time()-start,end_epoch=time.time(),completed_conditions=done);dump('receipt.json',receipt);signal.alarm(0)
if __name__=='__main__':main()
