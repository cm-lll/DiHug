"""Fixed manifest step128 EMA; inherited reverse kernel, no sampling changes."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[key]='1'
import sys,json,time,signal,traceback,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def write(path,obj):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False));tmp.replace(path)
def main():
    out=Path(sys.argv[1]);mi=int(sys.argv[2]);readout=sys.argv[3];mode=sys.argv[4]
    p=json.loads((out/'protocol.json').read_text());row=p['parents'][mi];selected_arm=sys.argv[5]
    folder=out/row['id']/selected_arm/readout;folder.mkdir(parents=True,exist_ok=True)
    start=time.time();steps=0;receipt=dict(start_epoch=start,device=os.environ['CUDA_VISIBLE_DEVICES'],mode=mode,status='running')
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(TimeoutError('owned watchdog stop')))
    def check():
        if time.time()>=p['science_stop_epoch'] or (out/'STOP').exists():raise TimeoutError('science/budget stop')
    spec=importlib.util.spec_from_file_location('support',ROOT/'research_branches/one_step_vs_terminal_plan_20261111/update_preflight.py')
    s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s);w,np,torch=s.w,s.np,s.torch
    try:
        check();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest')
        for path,h in p['runtime_source_bindings'].items():assert s.base.sha(path)==h,path
        data=np.load(row['data']);e=data['edges'];e=e if e.shape[0]==2 else e.T
        case=w.common.make_case(data['types'],e,tuple(map(tuple,data['schema'])))
        m=w.StructuralDenoiser(case,row['config'],'A').cuda();m.eval()
        write(folder/f'environment_{mode}.json',w.common.environment('cuda'))
        ids,fam,q=w.common.plan_for(case,2026111402,1).block(0)
        tasklist=[t for t in p['tasks'] if t['model']==row['id'] and t['readout']==readout and t['arm']==selected_arm]
        assert len(tasklist)==4
        if mode=='preflight':tasklist=tasklist[:1]
        for task in tasklist:
            check();arm=task['arm'];ss=task['sample_seed'];directory=folder/f'{arm}_seed{ss}';directory.mkdir(exist_ok=True)
            if (directory/'summary.json').exists():raise RuntimeError('refuse duplicate completed chain')
            cp=next(c for c in p['checkpoints'] if c['model']==row['id'] and c['arm']==arm and c['readout']==readout)
            assert s.base.sha(cp['checkpoint'])==cp['checkpoint_sha256']
            state=torch.load(cp['checkpoint'],map_location='cpu',weights_only=False)
            assert state['updates']==state['ema']['updates']==task['feedback_updates']==128 and state['ema']['decay']==.99
            effective=dict(state['model'])
            if readout=='ema':effective.update(state['ema']['shadow'])
            assert s.state_digest(effective)==cp['model_hash']==task['model_hash']
            m.load_state_dict(effective,strict=True);m.eval()
            assert s.state_digest(m.state_dict())==task['model_hash']
            parent=Path(task['paired_parent']);pi=json.loads((parent/'identity.json').read_text())
            oldsteps=[json.loads(x) for x in (parent/'steps.jsonl').read_text().splitlines()]
            assert pi['model_hash']==row['parent_model_hash'] and len(oldsteps)==1000
            generator=s.RecordedGenerator(ss,case['space'].total)
            families=np.repeat(np.arange(case['n_families']),case['space'].counts)
            x=(generator.random(case['space'].total)<case['prior'][families]).astype(np.uint8)
            assert s.digest(x)==pi['initial_hash']
            chain=w.SamplingBoundary(x,1000,generator,'cuda');kernel=w.MarginalKernel(case['prior'],w.common.schedule(1000))
            ident=dict(dataset=row['dataset'],train_seed=row['train_seed'],arm=arm,readout=readout,
                sample_seed=ss,model_hash=task['model_hash'],checkpoint_sha256=cp['checkpoint_sha256'],
                updates=task['feedback_updates'],T=1000,R=1,initial_hash=s.digest(x),protocol_sha256=s.base.sha(out/'protocol.json'))
            savedpath=directory/'latest_complete.pt'
            if savedpath.exists():
                assert json.loads((directory/'identity.json').read_text())==ident
                saved=torch.load(savedpath,map_location='cpu',weights_only=False);chain.restore(saved,ident)
                assert s.state_digest(chain.checkpoint(ident))==s.state_digest(saved)
            else:write(directory/'identity.json',ident)
            original_plan=w.common.plan_for;plans=[]
            def observed(*a,**kw):
                plan=s.RecordedPlan(original_plan(*a,**kw));plans.append(plan);return plan
            w.common.plan_for=observed;start_t=chain.next_t;tick=time.time()
            try:
                while chain.next_t>0:
                    check();t=chain.next_t;plans.clear();draws=generator.calls
                    expected=oldsteps[1000-t];assert expected['time']==t
                    def validate(candidate,record):
                        assert generator.calls==draws+1 and len(plans)==1 and plans[0].blocks==1
                        assert record['exact_coverage'] and record['covered_slots']==len(ids)==expected['covered_slots']
                        assert not np.any(candidate[~np.repeat(case['active'],case['space'].counts)])
                        assert generator.last_sha==expected['uniform_hash']
                        assert plans[0].h.hexdigest()==expected['query_hash']
                    rec=chain.advance(lambda a,b:w.runtime.reverse_step(m,case,a,kernel,b,1,ss,generator,'cuda',check_time=check),validate)
                    steps+=1;rec.update(uniform_hash=generator.last_sha,query_hash=plans[0].h.hexdigest(),paired_parent_contract=True)
                    with (directory/'steps.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
                    if t%100==0 or t==1 or (mode=='preflight' and start_t-chain.next_t==10):
                        tmp=savedpath.with_suffix('.tmp');torch.save(chain.checkpoint(ident),tmp);tmp.replace(savedpath)
                        write(folder/'progress.json',dict(epoch=time.time(),arm=arm,seed=ss,next_t=chain.next_t,steps_this_process=steps))
                    if mode=='preflight' and start_t-chain.next_t==10:break
                elapsed=time.time()-tick
                with (directory/'segments.jsonl').open('a') as f:f.write(json.dumps(dict(start_t=start_t,end_t=chain.next_t,seconds=elapsed))+'\n')
                assert s.state_digest(m.state_dict())==task['model_hash']
                saved=torch.load(savedpath,map_location='cpu',weights_only=False);chain.restore(saved,ident)
                assert s.state_digest(chain.checkpoint(ident))==s.state_digest(saved)
                if chain.next_t==0:
                    ee,ff=w.common.state_edges(case,chain.state)
                    np.savez_compressed(directory/'graph.npz',types=case['space'].types,edges=ee,families=ff,schema=np.array(case['space'].pairs))
                    lines=(directory/'steps.jsonl').read_text().splitlines();assert len(lines)==1000 and generator.calls==1001
                    seconds=sum(json.loads(l)['seconds'] for l in (directory/'segments.jsonl').read_text().splitlines())
                    write(directory/'summary.json',dict(status='complete',seconds=seconds,steps=1000,model_unchanged=True,
                        graph_sha256=s.base.sha(directory/'graph.npz'),paired_initial_query_uniform=True,checkpoint_unchanged=s.base.sha(cp['checkpoint'])==cp['checkpoint_sha256']))
                else:
                    assert mode=='preflight' and chain.next_t==990
                    write(folder/'calibration.json',dict(seconds=elapsed,steps=10,seconds_per_step=elapsed/10,
                        exact_serialized_boundary=True,paired_parent_contract=True,model_hash=task['model_hash'],
                        part_of_fixed_panel=True,CUDA_recalculation_bitwise_certified=False))
            except BaseException:
                try:torch.save(chain.checkpoint(ident),directory/'stopped_complete.pt')
                except BaseException:write(directory/'not_resumable.json',dict(traceback=traceback.format_exc()))
                raise
            finally:w.common.plan_for=original_plan
        receipt['status']='complete'
    except BaseException:
        receipt.update(status='failed',traceback=traceback.format_exc());raise
    finally:
        receipt.update(end_epoch=time.time(),seconds=time.time()-start,sampling_steps=steps,optimizer_updates=0,
            peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20 if torch.cuda.is_initialized() else 0)
        write(folder/f'receipt_{mode}.json',receipt)
if __name__=='__main__':main()
