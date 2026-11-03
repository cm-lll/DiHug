"""Review-gated single-GPU worker; not executed on scientific models in this ticket.
Initial/mid/final saves, paired normal2 inputs, final-only original full sampling.
"""
import argparse,pathlib,json,time,os,copy,signal,traceback,resource
import numpy as np
import torch
from prepare import save,sha
from experiment_contract import read_dictionary,scientific_gate,source_binding,initialization,reset_training_streams
from role_dictionary import seed32
from normal2 import normal2,pair_inputs,sync
from probe import rng_get,rng_set,equal,input_hash
from hsgd.bw_preflight.common import cpu_tree,environment,schedule,check_prob_device,state_edges
from hsgd.computational import FrozenTypedContext
from hsgd.losses import supported_binary_ce
from hsgd.kernel import MarginalKernel
from hsgd.runtime import reverse_step
from evaluation import evaluate

def atomic_checkpoint(path,payload):
    path=pathlib.Path(path)
    if path.exists():raise FileExistsError('checkpoint conflict '+str(path))
    temp=path.with_suffix('.tmp');torch.save(payload,temp)
    with temp.open('rb') as f:os.fsync(f.fileno())
    temp.replace(path);save(path.with_suffix('.sha256.json'),dict(sha256=sha(path),updates=payload.get('updates')))

def validation(model,case,cfg,seed,rng):
    before=rng_get(rng,'cuda');rows=[];start=time.perf_counter();mode=model.training;model.eval()
    try:
        for t in [1,500,1000]:
            private=np.random.default_rng(seed32(seed,purpose='fixed_validation_noise',time=t));inp=pair_inputs(case,cfg,10000+t,private,t)[0]
            ctx=FrozenTypedContext(model,case['space'].types,inp['edges'],inp['edge_families'],('validation',t))
            try:
                with torch.no_grad():
                    graph,logits,p=ctx.predict(inp['query'],inp['f'],t/cfg['T'],('validation',t),details=True)
                    check_prob_device(p,inp['f'],case,model.prior);labels=torch.as_tensor(inp['labels'],device='cuda');ce=supported_binary_ce(logits,labels,model.prior[graph.query_families]).cpu().numpy();prob=p.cpu().numpy()
                family=[]
                for f in range(case['n_families']):
                    ix=inp['f']==f;pos=ix&(inp['labels']==1);neg=ix&(inp['labels']==0)
                    family.append(dict(family=f,positive_count=int(pos.sum()),negative_count=int(neg.sum()),positive_ce=float(ce[pos].mean()) if pos.any() else None,negative_ce=float(ce[neg].mean()) if neg.any() else None,expected_edges=float(prob[ix].sum()),true_edges=int(pos.sum())))
                rows.append(dict(t=t,input_sha256=input_hash([inp]),ce_sum_over_S=float(ce.sum()/case['space'].total),family=family))
            finally:ctx.close()
    finally:model.train(mode);rng_set(before,rng,'cuda')
    if not equal(before,rng_get(rng,'cuda')):raise AssertionError('validation changed training RNG')
    return dict(seconds=time.perf_counter()-start,rows=rows,rng_unchanged=True,same_reference_only=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=['IMDB','PubMed'],required=True);ap.add_argument('--arm',choices=list('DE'),required=True);ap.add_argument('--train-seed',type=int,required=True);ap.add_argument('--output',required=True);ap.add_argument('--dictionary-dir',required=True);ap.add_argument('--ledger',required=True);ap.add_argument('--approval',required=True);ap.add_argument('--protocol',required=True);ap.add_argument('--cost',required=True);ap.add_argument('--launch-fd',type=int,required=True);args=ap.parse_args()
    from launch import gate
    from prepare import ROOT
    ticket=json.loads(os.read(args.launch_fd,16384));os.close(args.launch_fd)
    from runtime_plan import verify_ticket,initialization_for_task,task_id
    verify_ticket(ticket,args,os.getppid(),os.environ.get('CUDA_VISIBLE_DEVICES'),time.time())
    gate(ROOT,args.ledger,args.approval,args.protocol,args.dictionary_dir,args.cost)
    source,source_hash=source_binding();ledger,approval=scientific_gate(args.ledger,args.approval,args.protocol,args.dictionary_dir,source_hash)
    # Both datasets retain approved dictionaries; only E supplies the basis to the model.
    protocol=json.loads(pathlib.Path(args.protocol).read_text());dataset=args.dataset;arm=args.arm;out=pathlib.Path(args.output)
    if out.resolve()!=pathlib.Path(ledger['output']).resolve()/task_id(dataset,args.train_seed,arm):raise ValueError('task output not under activated run')
    out.mkdir(parents=True,exist_ok=False)
    stop=[False];in_update=False;completed=0;status=dict(status='initializing',formal_training_authorized=True,completed_updates=0,completed_chains=0)
    def requested(s,f):stop[0]=True
    for sig in [signal.SIGINT,signal.SIGTERM,signal.SIGHUP]:signal.signal(sig,requested)
    def check():
        if stop[0] or (out/'STOP').exists():raise InterruptedError('safe stop')
        if time.time()>=ledger['deadline_epoch']-1200:raise TimeoutError('evaluation/delivery reserve reached')
    try:
        check();torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True,warn_only=False);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or '2080 Ti' not in torch.cuda.get_device_name(0):raise RuntimeError('one frozen RTX2080Ti required')
        env=environment('cuda');save(out/'environment.json',env)
        # Numerical dependency and GPU-model equality is a required approval binding.
        for key in ['python_version','torch_runtime_version','device_name']:
            if approval['environment'][key]!=env[key]:raise ValueError('environment mismatch '+key)
        if approval['environment']['packages']!=env['packages']:raise ValueError('dependency versions differ')
        basis,receipt=read_dictionary(args.dictionary_dir,dataset,approval['dictionaries'][dataset]);seed=args.train_seed
        if seed not in protocol['train_seeds'][dataset]:raise ValueError('training seed not approved')
        model,case,cfg,proof=initialization_for_task(initialization,dataset,arm,seed,basis);cfg=dict(cfg,seed=seed)
        model=model.cuda();rng,rng_proof=reset_training_streams(seed,dataset,'cuda');save(out/'training_rng_reset.json',rng_proof)
        binding=dict(task_id=task_id(dataset,seed,arm),dataset_name=dataset,train_seed=seed,source_sha256=source_hash,source=source,protocol_sha256=sha(args.protocol),ledger_sha256=sha(args.ledger),dictionary_sha256=receipt['basis_sha256'],dataset=cfg['datasets'][dataset],arm=arm,config=cfg,environment=env,updates=approval['updates'])
        save(out/'binding.json',binding);save(out/'initialization.json',proof)
        # Entire initial state and RNG saved before any forward, isolated from engineering.
        initial=dict(model=cpu_tree(model.state_dict()),rng=rng_get(rng,'cuda'),updates=0,binding=binding,purpose='fresh scientific initial, never engineering weights')
        atomic_checkpoint(out/'initial.pt',initial);model=model.cuda();opt=torch.optim.AdamW(model.parameters(),**{k:v for k,v in cfg['optimizer'].items() if k!='type'})
        if {id(p) for g in opt.param_groups for p in g['params']}!={id(p) for p in model.parameters() if p.requires_grad}:raise AssertionError('optimizer membership')
        validation_rows=[dict(updates=0,**validation(model,case,cfg,seed,rng))];save(out/'validation.json',validation_rows)
        def snapshot(name):
            if in_update:raise RuntimeError('not an update boundary')
            atomic_checkpoint(out/name,dict(model=cpu_tree(model.state_dict()),optimizer=cpu_tree(opt.state_dict()),rng=rng_get(rng,'cuda'),updates=completed,binding=binding,elapsed_since_activation=time.time()-ledger['start_epoch']))
        target=approval['updates'];save(out/'progress.json',status)
        for position in range(1,target+1):
            check();inputs=pair_inputs(case,cfg,position,rng);in_update=True;rec=normal2(model,opt,case,inputs,1000,'cuda',True);in_update=False;completed=position
            rec.update(update=position,input_sha256=input_hash(inputs),t=[x['t'] for x in inputs],slots=[len(x['ids']) for x in inputs],utc=time.time(),warmup=position<=5)
            with (out/'training.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
            status.update(status='training',completed_updates=completed,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20);save(out/'progress.json',status)
            if position in [target//2,target]:
                snapshot('midpoint.pt' if position==target//2 else 'final.pt');validation_rows.append(dict(updates=position,**validation(model,case,cfg,seed,rng)));save(out/'validation.json',validation_rows)
        # Only the final weights are evaluated. Original runtime performs atomic full sweeps.
        model.eval();kernel=MarginalKernel(case['prior'],schedule(1000));families=np.repeat(np.arange(case['n_families']),case['space'].counts)
        next_t=None
        for sample_seed in protocol['sample_seeds']:
            check();folder=out/('sample_'+str(sample_seed));folder.mkdir();sample_rng=np.random.default_rng(sample_seed)
            state=(sample_rng.random(case['space'].total)<case['prior'][families]).astype(np.uint8);initial_rng=copy.deepcopy(sample_rng.bit_generator.state);begin=time.perf_counter();next_t=1000
            for t in range(1000,0,-1):
                check();candidate,rec=reverse_step(model,case,state,kernel,t,1,sample_seed,sample_rng,'cuda',check_time=check,trace=False)
                state=candidate;next_t=t-1 # commit only after exact full query coverage
                with (folder/'steps.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
                if t%100==0 or t==1:
                    atomic_checkpoint(folder/('complete_t%04d.pt'%next_t),dict(state=state,next_t=next_t,rng=rng_get(sample_rng,'cuda'),binding=binding,sample_seed=sample_seed,model_sha256=sha(out/'final.pt')))
            edges,fam=state_edges(case,state);np.savez_compressed(folder/'graph.npz',types=case['space'].types,edges=edges,families=fam,schema=np.array(case['space'].pairs))
            save(folder/'metrics.json',evaluate(case,edges,cfg['datasets'][dataset]['type_names'],check));save(folder/'binding.json',dict(checkpoint_sha256=sha(out/'final.pt'),graph_sha256=sha(folder/'graph.npz'),metrics_sha256=sha(folder/'metrics.json'),seed=sample_seed,initial_rng=initial_rng,seconds=time.perf_counter()-begin,binding=binding))
            status['completed_chains']+=1
        status['status']='complete'
    except BaseException as exc:
        status.update(status='failed_or_incomplete',error=repr(exc),traceback=traceback.format_exc(),incomplete_update_not_saved=in_update)
        if 'next_t' in locals() and next_t is not None:
            try:atomic_checkpoint(folder/'stopped_chain.pt',dict(state=state,next_t=next_t,rng=rng_get(sample_rng,'cuda'),binding=binding,sample_seed=sample_seed,model_sha256=sha(out/'final.pt')))
            except BaseException as save_error:status['chain_save_error']=repr(save_error)
        # No partial optimizer state is presented as a completed update.
        if 'opt' in locals() and not in_update:
            try:snapshot('stopped_u%d.pt'%completed)
            except BaseException as save_error:status['stop_save_error']=repr(save_error)
    finally:
        status.update(completed_updates=completed,finished_epoch=time.time(),elapsed_since_activation=time.time()-ledger['start_epoch'],peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024);save(out/'summary.json',status)
    return 0 if status['status']=='complete' else 2
if __name__=='__main__':raise SystemExit(main())
