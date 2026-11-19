"""Four-parent, two-arm, 16-update bounded diagnostic. No reverse chains."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import sys,json,time,signal,subprocess,traceback,importlib.util,itertools
from pathlib import Path
from model_check import sha,ROOT,OLD,PANEL,TIMES
OUT=ROOT/'results/analytic_joint_short_20261113'
POINTS=[1,4,8,16]
def write(path,obj):
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False))
def support():
    spec=importlib.util.spec_from_file_location('support',ROOT/'research_branches/one_step_vs_terminal_plan_20261111/update_preflight.py')
    s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s);return s
def prepare():
    OUT.mkdir(exist_ok=False)
    p=json.loads((PANEL/'protocol.json').read_text());old=json.loads((ROOT/'results/analytic_joint_model_check_20261113/protocol.json').read_text())
    bindings=dict(old['bindings'])
    for file in [Path(__file__),Path(__file__).with_name('model_check.py'),Path(__file__).with_name('core.py')]:bindings[str(file)]=sha(file)
    assert all(sha(path)==h for path,h in bindings.items())
    jobs=[j for j in p['jobs'] if j['architecture']=='A'];assert len(jobs)==4
    s=support();np=s.np;w=s.w
    frozen=[]
    for index,job in enumerate(jobs):
        row=p['parents'][job['parent']];seed=row['train_seed'];ds=job['dataset'];folder=OUT/job['id'];folder.mkdir()
        z=np.load(row['data']);ee=z['edges'];ee=ee if ee.shape[0]==2 else ee.T
        case=w.common.make_case(z['types'],ee,tuple(map(tuple,z['schema'])))
        payload={};manifest=[]
        for split,steps,root in [('train',range(1,17),202611133301),('validation',[0],202611133302)]:
            for step in steps:
                for j,t in enumerate(TIMES):
                    for rep in range(1 if split=='train' else 2):
                        tag=f'{split}_{step}_{j}_{rep}'
                        inp=w.common.train_input(case,1000,1,seed,940000+step*12+j*2+rep,
                            np.random.default_rng(np.random.SeedSequence([root,seed,step,j,rep])),force_t=t)
                        manifest.append(dict(tag=tag,split=split,step=step,t=t,rep=rep,input_hash=s.input_hash([inp])))
                        for key,value in inp.items():payload[tag+'__'+key]=value
        train={r['input_hash'] for r in manifest if r['split']=='train'}
        valid={r['input_hash'] for r in manifest if r['split']=='validation'}
        assert len(train)==96 and len(valid)==12 and not train&valid
        np.savez_compressed(folder/'inputs.npz',**payload);write(folder/'inputs.json',manifest)
        for file in [folder/'inputs.npz',folder/'inputs.json']:bindings[str(file)]=sha(file)
        frozen.append(dict(id=job['id'],GPU=index,parent=row,dataset=ds,train_seed=seed,
            initial=str(PANEL/job['id']/(ds+'_T_eager')/'initial_complete.pt')))
    start=time.time()
    protocol=dict(start_epoch=start,science_stop_epoch=start+1800,delivery_epoch=start+2100,
        max_card_seconds=1200,max_total_GPU_seconds=4800,max_GPU_hours=4/3,jobs=frozen,bindings=bindings,
        arms=['bias','risk'],updates_per_arm=16,temporary_updates=128,times=TIMES,points=POINTS,
        train_root=202611133301,validation_root=202611133302,validation_per_time=2,
        forwards=1296,backwards=768,per_worker_forwards=324,per_worker_backwards=192,
        input_rule='six fixed times per update, new independent forward noise each step; paired across arms; not IID t production training',
        loss='equal six-input mean, unscaled frozen bias/risk, no CE mix, no frequency/scale search',
        optimizer='restore sealed feedback step0, empty AdamW; lr1e-5 clip1 EMA.99; each arm evolves moments continuously',
        readout='raw at0/1/4/8/16, EMA at16, same fixed12 validation inputs; monitoring not untouched final test',
        scientific_boundary='conditional clean-prediction metrics only, no reverse chains; endpoint fixed16, no best checkpoint selection',
        decision='report both arms and all seeds; assess accumulated joint allocation improvement alongside low/hub/positive-edge/closure costs; no automatic longer training or generation',
        budget_basis='previous192F/48B/8updates332s singleGPU; new perGPU324F/192B+CPU scores/state checks estimated480-720s; cap1200s/card; four workers no waiting reservations',
        prior_gate='exact64graph formula and chain-rule check passed; four-parent one-step restore/RNG/finite updates passed',
        failures='stop affected worker; save last complete boundary if certified; never extend card/wall budget; keep all partial results')
    write(OUT/'protocol.json',protocol)
    print(json.dumps(dict(status='prepared',start_epoch=start,science_stop_epoch=protocol['science_stop_epoch'],delivery_epoch=protocol['delivery_epoch'],jobs=[j['id'] for j in frozen],bound_files=len(bindings))))

def worker(index):
    visible=os.environ['CUDA_VISIBLE_DEVICES'];from core import risks,probability_metrics
    os.environ['CUDA_VISIBLE_DEVICES']=visible;s=support();w,np,torch=s.w,s.np,s.torch
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    protocol=json.loads((OUT/'protocol.json').read_text());job=protocol['jobs'][index];folder=OUT/job['id']
    allocation=json.loads((folder/'allocation.json').read_text());stop=min(protocol['science_stop_epoch'],allocation['start_epoch']+protocol['max_card_seconds']-5)
    calls=backs=0;done=[];boundary=None;arm=None;result={}
    def check():
        if time.time()>=stop:raise TimeoutError('science or card limit')
    try:
        assert all(sha(path)==h for path,h in protocol['bindings'].items())
        write(folder/'environment.json',w.common.environment('cuda'))
        row=job['parent'];ds=job['dataset'];seed=job['train_seed']
        state=torch.load(job['initial'],map_location='cpu',weights_only=False)
        parent=torch.load(row['readout'],map_location='cpu',weights_only=False);cfg=parent['binding']['config']
        assert state['updates']==0 and not state['optimizer']['state']
        assert s.state_digest(state['model'])==s.state_digest(parent['model'])
        z=np.load(row['data']);types=z['types'];pairs=z['schema'];ee=z['edges'];ee=ee if ee.shape[0]==2 else ee.T
        case=w.common.make_case(types,ee,tuple(map(tuple,pairs)))
        q=np.array(list(itertools.combinations(range(len(types)),2))).T;lookup={tuple(v):i for i,v in enumerate(pairs)}
        f=np.array([lookup[tuple(sorted(types[e]))] for e in q.T]);keymap={tuple(e):i for i,e in enumerate(q.T)}
        refset={tuple(sorted(e)) for e in ee.T};ref=np.array([tuple(e) in refset for e in q.T],float)
        prior=np.bincount(f,weights=ref,minlength=len(pairs))/np.bincount(f,minlength=len(pairs))
        frozen=json.loads((OLD/(ds+'_frozen_target.json')).read_text())
        packed=np.load(folder/'inputs.npz');manifest=json.loads((folder/'inputs.json').read_text());inputs=[]
        for meta in manifest:
            prefix=meta['tag']+'__';inp={key[len(prefix):]:packed[key] for key in packed.files if key.startswith(prefix)}
            for k in list(inp):
                if inp[k].ndim==0:inp[k]=inp[k].item()
            assert s.input_hash([inp])==meta['input_hash'],meta['tag']
            inputs.append((meta,inp))
        m=w.StructuralDenoiser(case,cfg,'A').cuda();m.eval()
        oc={k:v for k,v in cfg['optimizer'].items() if k!='type'};oc['lr']=1e-5
        o=torch.optim.AdamW(m.parameters(),**oc);ema=s.ParameterEMA(m,.99);rng=np.random.default_rng()
        def restore():
            s.restore_payload(state,m,o,ema,state['binding'],lambda x:s.rng_restore(x,rng,'cuda'))
            o.zero_grad(set_to_none=True);m.eval();b=s.TrainingBoundary(m,o,ema,rng,'cuda',0)
            assert s.state_digest(b.checkpoint(state['binding']))==s.state_digest(state)
            return b
        boundary=restore()
        def forward(inp):
            nonlocal calls
            check();logits,prob=w.runtime.predict(m,types,inp['edges'],inp['edge_families'],inp['query'],inp['f'],inp['t']/1000,details=True);calls+=1
            w.common.check_prob_device(prob,inp['f'],case,m.prior);return logits,prob
        def expand(prob,inp,grad=False):
            ix=np.array([keymap[tuple(sorted(e))] for e in inp['query'].T]);assert len(set(ix))==len(ix)
            np.testing.assert_array_equal(f[ix],inp['f']);np.testing.assert_array_equal(ref[ix],inp['labels'])
            p=prior[f].copy();p[ix]=prob.detach().cpu().double().numpy();missing=np.ones(len(p),bool);missing[ix]=False
            assert np.all((p[missing]==0)|(p[missing]==1))
            return torch.tensor(p,dtype=torch.float64,requires_grad=grad),ix
        def evaluate(tag,ema_readout=False):
            before=s.state_digest(boundary.checkpoint(state['binding']));rows=[];dump={}
            def run_eval(model):
                with torch.no_grad():
                    for meta,inp in inputs:
                        if meta['split']!='validation':continue
                        logits,prob=forward(inp);tp,ix=expand(prob,inp);val,_=risks(tp,types,q,f,frozen)
                        pm=probability_metrics(tp.numpy(),types,q,f,ref,len(pairs));lab=inp['labels'];fam=inp['f'];pp=prob.cpu().double().numpy()
                        ce=s.supported_binary_ce(logits,torch.as_tensor(lab,device='cuda'),m.prior[torch.as_tensor(fam,device='cuda')]).cpu().numpy()
                        rel=[]
                        for r in range(len(pairs)):
                            pos=(fam==r)&(lab==1);neg=(fam==r)&(lab==0)
                            rel.append(dict(relation=r,positive_n=int(pos.sum()),negative_n=int(neg.sum()),
                                positive_CE=float(ce[pos].mean()) if pos.any() else None,negative_CE=float(ce[neg].mean()) if neg.any() else None,
                                missed_mass=float((1-pp[pos]).sum()),false_mass=float(pp[neg].sum())))
                        rows.append(dict(t=meta['t'],rep=meta['rep'],input_hash=meta['input_hash'],loss={k:float(v) for k,v in val.items()},
                            structure=pm,CE=float(ce.sum()/case['space'].total),positive_CE=float(ce[lab==1].mean()),negative_CE=float(ce[lab==0].mean()),relations=rel))
                        dump[meta['tag']]=tp.numpy()
            with boundary.validation():
                if ema_readout:s.ema_validation(run_eval,m,ema,boundary=boundary)
                else:run_eval(m)
            assert before==s.state_digest(boundary.checkpoint(state['binding']))
            np.savez_compressed(folder/(tag+'_probabilities.npz'),**dump,types=types,query=q,family=f,reference=ref)
            return rows
        result=dict(dataset=ds,train_seed=seed,baseline=evaluate('baseline'),arms={},serialized_restore_exact=True,CUDA_recomputation_bitwise_certified=False)
        write(folder/'result.json',result)
        for arm in ['bias','risk']:
            boundary=restore();record=dict(updates=[],evaluations={});result['arms'][arm]=record
            for step in range(1,17):
                check();tick=time.time();o.zero_grad(set_to_none=True);oldparam={n:v.detach().cpu().clone() for n,v in m.named_parameters()};terms=[];used=[]
                with boundary.step(step):
                    rng_before=s.state_digest(s.rng_capture(rng,'cuda'))
                    for meta,inp in inputs:
                        if meta['split']!='train' or meta['step']!=step:continue
                        logits,prob=forward(inp);tp,ix=expand(prob,inp,True);val,_=risks(tp,types,q,f,frozen)
                        derivative=torch.autograd.grad(val[arm]/6,tp)[0];assert torch.isfinite(derivative).all()
                        prob.backward(derivative[ix].to(device=prob.device,dtype=prob.dtype));backs+=1
                        terms.append(dict(t=meta['t'],input_hash=meta['input_hash'],**{k:float(v.detach()) for k,v in val.items()}));used.append(meta['input_hash'])
                    assert len(used)==6 and rng_before==s.state_digest(s.rng_capture(rng,'cuda'))
                    norm=float(torch.nn.utils.clip_grad_norm_(list(m.parameters()),1,error_if_nonfinite=True))
                    boundary.mark_optimizer_started();o.step();ema.update(m,step);ema.at_update_boundary=True
                    assert all(torch.isfinite(v).all() for v in m.parameters())
                delta=sum(float((v.detach().cpu().double()-oldparam[n].double()).square().sum()) for n,v in m.named_parameters())**.5
                record['updates'].append(dict(step=step,loss_inputs=terms,gradient_norm=norm,clip_scale=min(1,1/(norm+1e-6)),parameter_delta_L2=delta,seconds=time.time()-tick))
                if step in POINTS:
                    saved=boundary.checkpoint(state['binding']);assert saved['updates']==step and ema.updates==step
                    assert s.state_digest(saved['rng'])==s.state_digest(state['rng'])
                    cp=folder/f'{arm}_step{step:02d}_complete.pt';torch.save(saved,cp)
                    loaded=torch.load(cp,map_location='cpu',weights_only=False);assert s.state_digest(loaded)==s.state_digest(saved)
                    s.restore_payload(loaded,m,o,ema,state['binding'],lambda x:s.rng_restore(x,rng,'cuda'));m.eval()
                    assert s.state_digest(boundary.checkpoint(state['binding']))==s.state_digest(saved)
                    record['evaluations'][str(step)]=dict(raw=evaluate(f'{arm}_step{step:02d}_raw'),checkpoint_sha256=sha(cp))
                    if step==16:record['evaluations'][str(step)]['ema']=evaluate(f'{arm}_step{step:02d}_ema',True)
                write(folder/'result.json',result);write(folder/'progress.json',dict(arm=arm,step=step,forwards=calls,backwards=backs,completed_arms=done))
            done.append(arm)
        assert calls==324 and backs==192
        a=result['arms']['bias']['updates'];b=result['arms']['risk']['updates']
        assert [[x['input_hash'] for x in r['loss_inputs']] for r in a]==[[x['input_hash'] for x in r['loss_inputs']] for r in b]
        assert all(sha(path)==h for path,h in protocol['bindings'].items())
        write(folder/'completion.json',dict(status='complete',forwards=calls,backwards=backs,updates=32,new_reverse_chains=0,
            input_pairing=True,original_assets_unchanged=True,peak_GPU_MiB=torch.cuda.max_memory_allocated()/2**20))
    except BaseException:
        error=traceback.format_exc();rescue=None
        if boundary is not None:
            try:
                rescue=folder/'failure_last_complete.pt';torch.save(boundary.checkpoint(state['binding']),rescue)
            except BaseException:rescue=None
        write(folder/'failure.json',dict(error=error,arm=arm,done=done,forwards=calls,backwards=backs,
            recovery_checkpoint=str(rescue) if rescue else None));raise

def launch():
    if (OUT/'launch.json').exists():raise RuntimeError('duplicate launch refused')
    p=json.loads((OUT/'protocol.json').read_text());assert time.time()<p['science_stop_epoch']-1000
    assert all(sha(path)==h for path,h in p['bindings'].items())
    q=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    cards={int(r.split(', ')[0]):r.split(', ') for r in q.stdout.strip().splitlines()}
    for j in p['jobs']:
        c=cards[j['GPU']];assert int(c[2])==0 and int(c[3])<=5,(j['GPU'],c)
    write(OUT/'launch.json',dict(epoch=time.time(),resources=q.stdout,science_stop_epoch=p['science_stop_epoch'],delivery_epoch=p['delivery_epoch']))
    workers=[];total=0
    try:
        for i,job in enumerate(p['jobs']):
            folder=OUT/job['id'];start=time.time();env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(job['GPU'])
            write(folder/'allocation.json',dict(start_epoch=start,GPU=job['GPU'],uuid=cards[job['GPU']][1]))
            log=(folder/'worker.log').open('w');proc=subprocess.Popen([sys.executable,__file__,'worker',str(i)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            workers.append(dict(job=job,proc=proc,start=start,log=log,finished=False));write(folder/'pid.json',dict(pid=proc.pid))
        while not all(w['finished'] for w in workers):
            now=time.time();usage=total+sum(now-w['start'] for w in workers if not w['finished'])
            for w in workers:
                if w['finished']:continue
                proc=w['proc'];expired=now>=p['science_stop_epoch']-5 or now-w['start']>=p['max_card_seconds']-5 or usage>=p['max_total_GPU_seconds']-20
                if expired and proc.poll() is None:
                    os.killpg(proc.pid,signal.SIGTERM)
                    try:proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                if proc.poll() is not None:
                    end=time.time();seconds=end-w['start'];total+=seconds;w['finished']=True;w['log'].close()
                    write(OUT/w['job']['id']/'release.json',dict(exit_code=proc.returncode,owned_process_exited=True,start_epoch=w['start'],end_epoch=end,GPU_seconds=seconds,GPU_hours=seconds/3600))
            time.sleep(1)
    finally:
        for w in workers:
            if w['proc'].poll() is None:
                os.killpg(w['proc'].pid,signal.SIGTERM)
                try:w['proc'].wait(timeout=2)
                except subprocess.TimeoutExpired:os.killpg(w['proc'].pid,signal.SIGKILL);w['proc'].wait()
            if not w['finished']:
                end=time.time();seconds=end-w['start'];total+=seconds;w['log'].close()
                write(OUT/w['job']['id']/'release.json',dict(exit_code=w['proc'].returncode,owned_process_exited=True,start_epoch=w['start'],end_epoch=end,GPU_seconds=seconds,GPU_hours=seconds/3600))
        write(OUT/'release.json',dict(GPU_hours=total/3600,all_owned_processes_exited=all(w['proc'].poll() is not None for w in workers),exit_codes=[w['proc'].returncode for w in workers]))
    if any(w['proc'].returncode for w in workers):raise SystemExit(1)
if __name__=='__main__':
    if sys.argv[1]=='worker':worker(int(sys.argv[2]))
    else:{'prepare':prepare,'launch':launch}[sys.argv[1]]()
