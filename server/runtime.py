"""Portable single-device adapter. Reuses frozen model/kernel/transaction code.

This adapter is new deployment code, not the entry that produced the published
tables. Large-graph throughput and CUDA replay must be checked on the target.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
import argparse, hashlib, json, random, signal, sys, time, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CUR=ROOT/'current'
sys.path[:0]=[str(CUR/'research_branches/new_datasets_20261112'),
              str(CUR/'research_branches/matched_block_training_20261112')]
import support as lib
import sweep
s,w,np,torch=lib.s,lib.w,lib.np,lib.torch

def write(path,value):
    lib.base.write(path,value)

def load_case(path):
    with np.load(path,allow_pickle=False) as z:
        types=z['types']; edges=z['edges']; schema=z['schema']
        if edges.ndim!=2:raise ValueError('edges must be [2,E] or [E,2]')
        if edges.shape[0]!=2:edges=edges.T
        if not np.issubdtype(types.dtype,np.integer) or not np.issubdtype(edges.dtype,np.integer):
            raise ValueError('integer type/edge indices required')
        if types.ndim!=1 or len(types)==0 or (types<0).any():raise ValueError('invalid types')
        if not np.array_equal(np.unique(types),np.arange(int(types.max())+1)):raise ValueError('contiguous present type IDs required')
        if schema.ndim!=2 or schema.shape[1]!=2 or len(set(map(tuple,schema)))!=len(schema):raise ValueError('unique type-pair schema required')
        if (schema[:,0]>schema[:,1]).any() or schema.min()<0 or schema.max()>types.max():raise ValueError('invalid ordered schema')
        if edges.size and ((edges<0).any() or (edges>=len(types)).any() or (edges[0]>=edges[1]).any()):raise ValueError('canonical undirected edges u<v required; no self loops')
        if len(set(map(tuple,edges.T)))!=edges.shape[1]:raise ValueError('duplicate edges')
        case=w.common.make_case(types,edges,tuple(map(tuple,schema)))
        if 'families' in z and not np.array_equal(case['space'].encode(edges)[0],z['families']):raise ValueError('family labels disagree with schema')
        if 'relations' in z and not np.array_equal(case['space'].encode(edges)[0]+1,z['relations']):raise ValueError('relation labels disagree with schema')
    return case

def ce_step(model,opt,ema,boundary,case,cfg,seed,rounds,rng,check):
    position=boundary.completed+1
    with boundary.step(position):
        inp=w.common.train_input(case,1000,rounds,seed,position,rng)
        model.train(); opt.zero_grad(set_to_none=True);loss_total=0.
        for ids,fam,logits,prob,plan in sweep.blocks(model,case,inp['edges'],inp['edge_families'],inp['t'],rounds,seed+position,check):
            y=torch.as_tensor(np.isin(ids,case['clean_ids']),device=logits.device)
            loss=s.supported_binary_ce(logits,y,model.prior[torch.as_tensor(fam,device=logits.device)]).sum()/case['space'].total
            if not torch.isfinite(loss):raise FloatingPointError('CE')
            loss.backward();loss_total+=float(loss.detach())
        commit(model,opt,ema,boundary,clip=False)
    return dict(update=position,t=inp['t'],CE=loss_total,input_hash=s.input_hash([inp]),query_hash=plan.h.hexdigest(),rounds=rounds)

def commit(model,opt,ema,boundary,clip):
    grads=[p.grad for p in model.parameters() if p.grad is not None]
    if not grads or not all(torch.isfinite(g).all() for g in grads):raise FloatingPointError('gradients')
    if clip:torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True)
    boundary.mark_optimizer_started();ema.at_update_boundary=False
    opt.step();ema.update(model,boundary.completed+1);ema.at_update_boundary=True
    if not all(torch.isfinite(p).all() for p in model.parameters()):raise FloatingPointError('parameters')

def sample(model,case,rounds,seed,dest,device,check,selected=()):
    dest.mkdir(exist_ok=False);model.eval()
    generator=s.RecordedGenerator(seed,case['space'].total)
    families=np.repeat(np.arange(case['n_families']),case['space'].counts)
    state=(generator.random(case['space'].total)<case['prior'][families]).astype(np.uint8)
    chain=w.SamplingBoundary(state,1000,generator,device)
    kernel=w.MarginalKernel(case['prior'],w.common.schedule(1000));snapshots=[]
    ident=dict(sample_seed=seed,rounds=rounds,T=1000,model_hash=s.state_digest(model.state_dict()))
    write(dest/'identity.json',ident)
    original=w.common.plan_for;plans=[]
    def plan(*a,**kw):
        p=s.RecordedPlan(original(*a,**kw));plans.append(p);return p
    w.common.plan_for=plan
    try:
        for t in range(1000,0,-1):
            check();plans.clear()
            if t in selected:
                path=dest/f'Xt_{t:04d}.npz';np.savez_compressed(path,xt=chain.state,t=t);snapshots.append(path)
            def validate(candidate,record):
                assert record['exact_coverage'] and len(plans)==1 and plans[0].blocks==rounds
                assert not candidate[~np.repeat(case['active'],case['space'].counts)].any()
            rec=chain.advance(lambda x,at:w.runtime.reverse_step(model,case,x,kernel,at,rounds,seed,generator,device,check_time=check),validate)
            rec.update(uniform_hash=generator.last_sha,query_hash=plans[0].h.hexdigest())
            with (dest/'steps.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
            if t%100==0:torch.save(chain.checkpoint(ident),dest/'latest_complete.pt')
        edges,fam=w.common.state_edges(case,chain.state)
        np.savez_compressed(dest/'graph.npz',types=case['space'].types,edges=edges,families=fam,schema=np.array(case['space'].pairs))
        assert chain.next_t==0 and generator.calls==1001
        write(dest/'summary.json',dict(status='complete',steps=1000,graph_sha256=lib.base.sha(dest/'graph.npz')))
        return chain.state.copy(),snapshots
    except BaseException:
        try:torch.save(chain.checkpoint(ident),dest/'stopped_complete.pt')
        except BaseException:write(dest/'nonresumable.json',dict(use_previous_complete=True))
        raise
    finally:w.common.plan_for=original

def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('mode',choices=['smoke','train-ce','sample','feedback'])
    a.add_argument('--data',type=Path,required=True);a.add_argument('--config',type=Path,required=True)
    a.add_argument('--out',type=Path,required=True);a.add_argument('--arm',choices=['O','A'],default='A')
    a.add_argument('--seed',type=int,required=True);a.add_argument('--rounds',type=int,required=True)
    a.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    a.add_argument('--updates',type=int,default=None);a.add_argument('--minutes',type=float,required=True)
    a.add_argument('--checkpoint',type=Path);a.add_argument('--checkpoint-sha256')
    a.add_argument('--resume',action='store_true',help='Restore this adapter full state; --updates is the total target')
    a.add_argument('--readout',choices=['raw','ema'],default='raw');a.add_argument('--sample-seeds',type=int,nargs='+',default=[2026102821,2026102822,2026110231,2026110232])
    a.add_argument('--allow-dense-score-nodes',type=int,default=2000)
    args=a.parse_args();start=time.time();deadline=start+args.minutes*60
    if args.minutes<=0 or args.rounds<1:a.error('positive budget and rounds required')
    if args.mode in ('train-ce','feedback') and (args.updates is None or args.updates<=0):a.error('explicit positive --updates required')
    if args.mode in ('sample','feedback') and not args.checkpoint:a.error('sample/feedback require a trusted checkpoint')
    if args.checkpoint and (not args.checkpoint_sha256 or lib.base.sha(args.checkpoint)!=args.checkpoint_sha256):a.error('checkpoint SHA256 is required and must match')
    if args.resume and (not args.checkpoint or args.mode not in ('train-ce','feedback') or args.readout!='raw'):a.error('resume requires a raw full training state')
    args.out.mkdir(parents=True,exist_ok=False)
    def check():
        if time.time()>=deadline:raise TimeoutError('explicit runtime deadline')
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(TimeoutError('external watchdog stop')))
    boundary=None
    try:
        lib.setup();case=load_case(args.data);cfg=json.loads(args.config.read_text());cfg['seed']=args.seed
        cfg['T']=1000;cfg['rounds']=args.rounds
        if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no silent CPU fallback')
        lib.seed_all(args.seed);model=w.StructuralDenoiser(case,cfg,args.arm).to(args.device)
        parent=None
        if args.checkpoint:
            parent=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
            # Portability is not permission to reuse a small graph's prior on a new graph.
            if parent.get('binding',{}).get('data_sha256') not in (None,lib.base.sha(args.data)):raise ValueError('checkpoint data identity mismatch')
            state=dict(parent['model'])
            if args.readout=='ema':
                if 'ema' in parent:state.update(parent['ema']['shadow'])
                elif parent.get('readout')!='ema_0.99':raise ValueError('no certified EMA readout')
            if not torch.equal(state['prior'].cpu(),model.prior.cpu()):raise ValueError('checkpoint prior differs; train a data-matched model')
            model.load_state_dict(state,strict=True)
        opt_cfg={k:v for k,v in cfg['optimizer'].items() if k!='type'}
        if args.mode=='feedback':opt_cfg['lr']=1e-5
        opt=torch.optim.AdamW(model.parameters(),**opt_cfg);ema=s.ParameterEMA(model,.99)
        lib.seed_all(args.seed+1);rng=np.random.default_rng(np.random.SeedSequence([2026111202,args.seed]))
        boundary=s.TrainingBoundary(model,opt,ema,rng,args.device,0)
        binding=dict(data_sha256=lib.base.sha(args.data),config=cfg,arm=args.arm,train_seed=args.seed,mode=args.mode,rounds=args.rounds,parent_sha256=args.checkpoint_sha256,new_optimizer=True,new_EMA=True,adapter='server/runtime.py',scientific_source='current/',not_historical_bitwise_reproduction=True)
        if args.resume:
            old=parent['binding']
            for key in ('data_sha256','config','arm','train_seed','mode','rounds','adapter'):
                if old[key]!=binding[key]:raise ValueError('resume identity mismatch: '+key)
            binding=old
            s.restore_payload(parent,model,opt,ema,binding,lambda x:s.rng_restore(x,rng,args.device))
            boundary=s.TrainingBoundary(model,opt,ema,rng,args.device,parent['updates'])
            assert s.state_digest(boundary.checkpoint(binding))==s.state_digest(parent)
            if boundary.completed>=args.updates:raise ValueError('already at requested endpoint; no duplicate work')
        write(args.out/'binding.json',binding);write(args.out/'environment.json',w.common.environment(args.device))
        def save(name):
            temp=args.out/(name+'.tmp');torch.save(boundary.checkpoint(binding),temp);temp.replace(args.out/name)
        save('initial_complete.pt')
        if args.mode in ('train-ce','smoke'):
            for _ in range(1 if args.mode=='smoke' else args.updates-boundary.completed):
                check();rec=ce_step(model,opt,ema,boundary,case,cfg,args.seed,args.rounds,rng,check)
                with (args.out/'training.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
                if boundary.completed%100==0:save('latest_complete.pt')
            save('final_complete.pt')
            if args.mode=='smoke':
                z=torch.load(args.out/'final_complete.pt',map_location='cpu',weights_only=False)
                s.restore_payload(z,model,opt,ema,binding,lambda x:s.rng_restore(x,rng,args.device))
                restored=s.TrainingBoundary(model,opt,ema,rng,args.device,z['updates'])
                assert s.state_digest(restored.checkpoint(binding))==s.state_digest(z)
                # One reverse step, not an invented short final graph.
                gen=np.random.default_rng(args.seed+7);f=np.repeat(np.arange(case['n_families']),case['space'].counts)
                x=(gen.random(case['space'].total)<case['prior'][f]).astype(np.uint8)
                model.eval();_,rec=w.runtime.reverse_step(model,case,x,w.MarginalKernel(case['prior'],w.common.schedule(1000)),1000,args.rounds,args.seed+7,gen,args.device,check_time=check)
                assert rec['exact_coverage']
                write(args.out/'smoke.json',dict(saved_state_exact=True,reverse_step_complete=True,CUDA_recomputed_update_bitwise_not_certified=True))
        elif args.mode=='sample':
            for seed in args.sample_seeds:sample(model,case,args.rounds,seed,args.out/f'sample_{seed}',args.device,check)
        else:
            if len(case['space'].types)>args.allow_dense_score_nodes:raise ValueError('dense scoring needs explicit resource review; increase --allow-dense-score-nodes only after measurement')
            ids,fam,q=w.common.plan_for(case,2026111402,1).block(0);labels=np.isin(ids,case['clean_ids']).astype(np.uint8)
            scorer=lib.Score('server',case,q,fam,labels);write(args.out/'score_scaling.json',scorer.s)
            planning=np.random.default_rng(np.random.SeedSequence([2026111405,args.seed]))
            plans={str(u):[dict(seed=int(planning.integers(1,2**31-1)),times=sorted(map(int,planning.choice(np.arange(1,1001),8,replace=False)),reverse=True)) for _ in range(2)] for u in range(1,args.updates+1)}
            write(args.out/'terminal_plan.json',plans)
            for u in range(boundary.completed+1,args.updates+1):
                check();batch=[];save('latest_complete.pt')
                for p in plans[str(u)]:
                    full,paths=sample(model,case,args.rounds,p['seed'],args.out/f'train_{u}_{p["seed"]}',args.device,check,p['times'])
                    batch.append((full,paths,sum(scorer(full[ids])[:2]),p['seed']))
                costs=np.array([b[2] for b in batch]);adv=2*costs-costs.sum()
                model.eval();opt.zero_grad(set_to_none=True)
                with boundary.step(u):
                    for k,(target,paths,cost,seed) in enumerate(batch):
                        for path in paths:
                            with np.load(path) as z:edges,ef=w.common.state_edges(case,z['xt']);t=int(z['t'])
                            record=json.loads((path.parent/'steps.jsonl').read_text().splitlines()[1000-t])
                            sweep.feedback(model,case,edges,ef,t,args.rounds,seed,target,.01*float(adv[k])/16,record['query_hash'],check)
                    commit(model,opt,ema,boundary,clip=True)
                write(args.out/f'update_{u}.json',dict(costs=costs.tolist(),advantages=adv.tolist(),clean_head_proxy_is_biased=True))
                save('latest_complete.pt')
            save('final_complete.pt')
        write(args.out/'receipt.json',dict(status='complete',mode=args.mode,seconds=time.time()-start,updates=boundary.completed,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20 if args.device=='cuda' else 0,full_graph_quality_not_certified=True))
    except BaseException:
        saved=False
        if boundary is not None and boundary.resumable and boundary.phase=='idle':
            try:torch.save(boundary.checkpoint(binding),args.out/'stopped_complete.pt');saved=True
            except BaseException:pass
        write(args.out/'failure.json',dict(traceback=traceback.format_exc(),resumable_state_saved=saved,use_previous_complete_if_false=True));raise
if __name__=='__main__':main()
