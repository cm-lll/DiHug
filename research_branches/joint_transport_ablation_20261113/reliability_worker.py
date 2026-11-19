"""Frozen-policy reliability collection. Never calls optimizer.step or EMA.update.

All sampling uses the existing transactional reverse kernel. Gradients are the
existing eager clean-head surrogate, not unbiased trajectory reward gradients.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
import sys,json,time,signal,traceback,hashlib,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def write(path,obj):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False));tmp.replace(path)

def main():
    out=Path(sys.argv[1]);mi=int(sys.argv[2]);half=int(sys.argv[3]);mode=sys.argv[4]
    protocol=json.loads((out/'protocol.json').read_text());row=protocol['models'][mi]
    folder=out/row['id'];folder.mkdir(exist_ok=True)
    job=f'{mi}_{half}_{mode}';start=time.time();calls=backs=steps=0
    receipt=dict(job=job,status='running',start_epoch=start,device=os.environ['CUDA_VISIBLE_DEVICES'])
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(TimeoutError('owned budget watchdog')))
    def check():
        if time.time()>=protocol['science_stop_epoch'] or (out/'STOP').exists():
            raise TimeoutError('science/card-hour stop')
    spec=importlib.util.spec_from_file_location('check_support',ROOT/'research_branches/one_step_vs_terminal_plan_20261111/update_preflight.py')
    s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
    w,np,torch=s.w,s.np,s.torch
    try:
        check();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest')
        for path,h in protocol['source_bindings'].items():assert s.base.sha(path)==h,path
        cfg=row['config'];z=np.load(row['data']);ee=z['edges'];ee=ee if ee.shape[0]==2 else ee.T
        case=w.common.make_case(z['types'],ee,tuple(map(tuple,z['schema'])))
        state=torch.load(row['initial_complete'],map_location='cpu',weights_only=False)
        assert state['updates']==0 and not state['optimizer']['state']
        m=w.StructuralDenoiser(case,cfg,'A').cuda();m.eval()
        oc={k:v for k,v in cfg['optimizer'].items() if k!='type'};oc['lr']=1e-5
        optim=torch.optim.AdamW(m.parameters(),**oc);ema=s.ParameterEMA(m,.99);rng=np.random.default_rng()
        s.restore_payload(state,m,optim,ema,state['binding'],lambda x:s.rng_restore(x,rng,'cuda'))
        boundary=s.TrainingBoundary(m,optim,ema,rng,'cuda',0)
        original_hash=s.state_digest(state)
        assert s.state_digest(boundary.checkpoint(state['binding']))==original_hash
        assert s.state_digest(m.state_dict())==row['model_hash']
        write(folder/f'environment_{job}.json',w.common.environment('cuda'))
        ids,fam,query=w.common.plan_for(case,2026111402,1).block(0)
        layout=[];offset=0
        for name,p in m.named_parameters():
            group='CN' if name.startswith('common_neighbor.') else 'joint_head' if name.startswith('joint_head.') else 'degree' if name.startswith('branch.') else 'backbone'
            layout.append(dict(name=name,start=offset,end=offset+p.numel(),group=group,shape=list(p.shape)))
            offset+=p.numel()
        write(folder/f'layout_{job}.json',layout)
        def unchanged():
            assert original_hash==s.state_digest(boundary.checkpoint(state['binding']))
        def gradient(plan,directory):
            nonlocal calls,backs
            index=plan['index'];vp=folder/f'vector_{index:02d}.npy';rp=folder/f'vector_{index:02d}.json'
            if vp.exists():
                info=json.loads(rp.read_text());assert info['complete'] and info['model_hash']==row['model_hash']
                assert s.base.sha(vp)==info['sha256'];return
            check();tick=time.time();optim.zero_grad(set_to_none=True)
            g=np.load(directory/'graph.npz');edges=g['edges'];edges=edges if edges.shape[0]==2 else edges.T
            eset={tuple(sorted(map(int,e))) for e in edges.T}
            target=torch.as_tensor([tuple(sorted(map(int,e))) in eset for e in query.T],device='cuda').bool()
            inputs=[]
            for t in plan['times']:
                check();xp=directory/f'Xt_{t:04d}.npz';z=np.load(xp);assert int(z['t'])==t
                ee,ff=w.common.state_edges(case,z['xt'])
                logits,p=w.runtime.predict(m,case['space'].types,ee,ff,query,fam,t/1000,details=True);calls+=1
                w.common.check_prob_device(p,fam,case,m.prior)
                lp=torch.log_softmax(logits.double(),-1)
                loss=torch.where(target,lp[:,1],lp[:,0]).sum()/len(plan['times'])
                assert torch.isfinite(loss);loss.backward();backs+=1
                inputs.append(dict(t=t,sha256=s.base.sha(xp),log_probability=float(loss.detach().cpu())))
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
            vector=torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten() for p in m.parameters()]).detach().cpu().numpy()
            assert np.isfinite(vector).all() and len(vector)==offset
            unchanged();np.save(vp,vector)
            write(rp,dict(complete=True,index=index,model_hash=row['model_hash'],graph_sha256=s.base.sha(directory/'graph.npz'),
                          sha256=s.base.sha(vp),seconds=time.time()-tick,inputs=inputs,norm=float(np.linalg.norm(vector.astype('float64'))),
                          optimizer_updates=0,model_optimizer_EMA_RNG_unchanged=True))
            optim.zero_grad(set_to_none=True)
        def sample(plan,limit=None):
            nonlocal steps
            check();ss=plan['seed'];directory=folder/f'chain_{plan["index"]:02d}_{ss}';directory.mkdir(exist_ok=True)
            if (directory/'summary.json').exists():raise RuntimeError('refuse duplicate completed chain')
            generator=s.RecordedGenerator(ss,case['space'].total)
            families=np.repeat(np.arange(case['n_families']),case['space'].counts)
            x=(generator.random(case['space'].total)<case['prior'][families]).astype(np.uint8)
            chain=w.SamplingBoundary(x,1000,generator,'cuda');kernel=w.MarginalKernel(case['prior'],w.common.schedule(1000))
            ident=dict(dataset=row['dataset'],train_seed=row['train_seed'],sample_seed=ss,model_hash=row['model_hash'],updates=0,
                       T=1000,R=1,initial_hash=s.digest(x),protocol_sha256=s.base.sha(out/'protocol.json'))
            if (directory/'latest_complete.pt').exists():
                assert json.loads((directory/'identity.json').read_text())==ident
                saved=torch.load(directory/'latest_complete.pt',map_location='cpu',weights_only=False)
                chain.restore(saved,ident)
                assert s.state_digest(chain.checkpoint(ident))==s.state_digest(saved)
            else:write(directory/'identity.json',ident)
            start_t=chain.next_t;tick=time.time();original_plan=w.common.plan_for;plans=[]
            def observed(*a,**kw):
                p=s.RecordedPlan(original_plan(*a,**kw));plans.append(p);return p
            w.common.plan_for=observed
            try:
                while chain.next_t>0:
                    check();t=chain.next_t;plans.clear();draws=generator.calls
                    if t in plan['times']:np.savez_compressed(directory/f'Xt_{t:04d}.npz',xt=chain.state,t=t)
                    def validate(candidate,record):
                        assert generator.calls==draws+1 and len(plans)==1 and plans[0].blocks==1
                        assert record['exact_coverage'] and record['covered_slots']==len(ids)
                        assert not np.any(candidate[~np.repeat(case['active'],case['space'].counts)])
                    rec=chain.advance(lambda a,b:w.runtime.reverse_step(m,case,a,kernel,b,1,ss,generator,'cuda',check_time=check),validate)
                    steps+=1;rec.update(uniform_hash=generator.last_sha,query_hash=plans[0].h.hexdigest())
                    with (directory/'steps.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
                    if t%100==0 or t==1 or (limit and start_t-chain.next_t>=limit):
                        torch.save(chain.checkpoint(ident),directory/'latest_complete.pt')
                        write(folder/f'progress_{job}.json',dict(epoch=time.time(),chain=plan['index'],next_t=chain.next_t,steps=steps,forwards=calls,backwards=backs))
                    if limit and start_t-chain.next_t>=limit:break
                elapsed=time.time()-tick
                with (directory/'segments.jsonl').open('a') as f:f.write(json.dumps(dict(start_t=start_t,end_t=chain.next_t,seconds=elapsed))+'\n')
                unchanged()
                if chain.next_t==0:
                    ee,ff=w.common.state_edges(case,chain.state)
                    np.savez_compressed(directory/'graph.npz',types=case['space'].types,edges=ee,families=ff,schema=np.array(case['space'].pairs))
                    lines=(directory/'steps.jsonl').read_text().splitlines();assert len(lines)==1000 and generator.calls==1001
                    for at,line in zip(range(1000,0,-1),lines):assert json.loads(line)['time']==at
                    assert all((directory/f'Xt_{t:04d}.npz').exists() for t in plan['times'])
                    secs=sum(json.loads(l)['seconds'] for l in (directory/'segments.jsonl').read_text().splitlines())
                    write(directory/'summary.json',dict(status='complete',seconds=secs,steps=1000,model_unchanged=True,graph_sha256=s.base.sha(directory/'graph.npz')))
                elif limit:
                    saved=torch.load(directory/'latest_complete.pt',map_location='cpu',weights_only=False)
                    chain.restore(saved,ident);assert s.state_digest(saved)==s.state_digest(chain.checkpoint(ident))
                    write(folder/f'calibration_{job}.json',dict(seconds=elapsed,steps=start_t-chain.next_t,seconds_per_step=elapsed/(start_t-chain.next_t),
                              exact_saved_boundary=True,resume_next_t=chain.next_t,new_chain_is_part_of_fixed40=True))
                return directory
            except BaseException:
                torch.save(chain.checkpoint(ident),directory/'stopped_complete.pt');raise
            finally:w.common.plan_for=original_plan
        if mode=='preflight':
            gradient(row['reused'][0],Path(row['reused'][0]['path']))
            sample(row['new_trajectories'][0],limit=10)
        else:
            assert mode=='run'
            selected=row['new_trajectories'][half*5:(half+1)*5]
            for plan in selected:sample(plan)
            for plan in row['reused'][half*3:(half+1)*3]:gradient(plan,Path(plan['path']))
            for plan in selected:gradient(plan,folder/f'chain_{plan["index"]:02d}_{plan["seed"]}')
        unchanged();receipt.update(status='complete',model_optimizer_EMA_RNG_unchanged=True)
    except BaseException:
        receipt.update(status='failed',traceback=traceback.format_exc());raise
    finally:
        receipt.update(end_epoch=time.time(),seconds=time.time()-start,sampling_steps=steps,gradient_forwards=calls,backwards=backs,
                       peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20 if torch.cuda.is_initialized() else 0)
        write(folder/f'receipt_{job}.json',receipt)

if __name__=='__main__':main()
