"""Bounded matched CPU analytic-loss / CUDA network VJP diagnostic.

Eight temporary updates, no chains. Original states and sources are read-only.
The exact loss is on CPU; its dL/dp is passed to the original GPU network.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import sys, json, time, hashlib, signal, subprocess, traceback, importlib.util, itertools
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/analytic_joint_model_check_20261113'
OLD = ROOT / 'results/analytic_joint_risk_20261113'
PANEL = ROOT / 'results/joint_transport_ablation_20261113T011928'
TIMES = [1,75,250,550,800,1000]

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''): h.update(b)
    return h.hexdigest()

def write(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False))

def prepare():
    OUT.mkdir(exist_ok=False)
    p=json.loads((PANEL/'protocol.json').read_text());bindings=dict(p['bindings'])
    for file in [Path(__file__), Path(__file__).with_name('core.py'),
                 ROOT/'research_branches/one_step_vs_terminal_plan_20261111/update_preflight.py']:
        bindings[str(file)]=sha(file)
    jobs=[]
    for job in p['jobs']:
        if job['architecture']!='A': continue
        row=p['parents'][job['parent']]
        files=[PANEL/job['id']/(job['dataset']+'_T_eager')/'initial_complete.pt',
               Path(row['readout']),Path(row['data']),OLD/(job['dataset']+'_frozen_target.json'),
               ROOT/'results/increment_update_check_20261113'/(job['id']+'.json')]
        for file in files: bindings[str(file)]=sha(file)
        jobs.append(job['id'])
    assert len(jobs)==4
    assert all(sha(path)==h for path,h in bindings.items())
    start=time.time()
    write('protocol.json',dict(start_epoch=start,delivery_epoch=start+1800,science_stop_epoch=start+1500,
        max_GPU_seconds=900,max_GPU_hours=.25,jobs=jobs,arms=['bias','risk'],updates=8,
        times=TIMES,update_input_root=202611130953,independent_readout_root=202611133211,
        old_inputs='six viewed forward-noise inputs per parent; reused for both arms, averaged equally',
        new_inputs='six new inputs per parent; disjoint RNG root, not used in update; same X0, not cross-graph validation',
        forwards=192,backwards=48,readout='raw; EMA saved but not evaluated',
        optimizer='original feedback initial_complete empty AdamW; lr1e-5; original clip1 and .99 EMA; no CE mixing',
        loss='unscaled mean across six times of frozen bias or exact risk; no loss/gradient scale search',
        purpose='single first-feedback-step transmission, not historical Adam continuation or CE-loss-only counterfactual',
        comparisons='unupdated parent and two matched temporary arms; no new reverse chains',
        budget_basis='historical 480F/384B/12updates occupied681s; 192F/48B/8updates expected300-600s with CPU analytic bridge; cap900s',
        numerical='original deterministic flags preserved; GPU repeated updates not certified bitwise',
        decision='report all seeds/times/relations; objective decrease alone is insufficient, hub/closure tradeoffs retained; no automatic formal training',
        bindings=bindings))
    print(json.dumps(dict(status='prepared',jobs=jobs,files=len(bindings),delivery_epoch=start+1800)))

def worker():
    # Historical CPU reference module sets CUDA visibility on import. Restore
    # the launcher binding before CUDA initialization; do not change that file.
    visible=os.environ['CUDA_VISIBLE_DEVICES']
    from core import risks, probability_metrics
    os.environ['CUDA_VISIBLE_DEVICES']=visible
    spec=importlib.util.spec_from_file_location('support',ROOT/'research_branches/one_step_vs_terminal_plan_20261111/update_preflight.py')
    s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
    w,np,torch=s.w,s.np,s.torch
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    proto=json.loads((OUT/'protocol.json').read_text());launch=json.loads((OUT/'launch.json').read_text())
    stop=min(proto['science_stop_epoch'],launch['allocation_start']+proto['max_GPU_seconds']-5)
    calls=backs=0;done=[]
    def check():
        if time.time()>=stop: raise TimeoutError('science/GPU limit')
    def group(name):
        return 'CN' if name.startswith('common_neighbor.') else 'joint_head' if name.startswith('joint_head.') else 'degree' if name.startswith('branch.') else 'backbone'
    def cosine(a,b):
        aa=sum(float(x.double().square().sum()) for x in a.values())
        bb=sum(float(x.double().square().sum()) for x in b.values())
        dot=sum(float((a[k].double()*b[k].double()).sum()) for k in a)
        return dict(left_norm=aa**.5,right_norm=bb**.5,cosine=dot/(aa*bb)**.5 if aa*bb else None)
    try:
        assert all(sha(path)==h for path,h in proto['bindings'].items())
        write('environment.json',w.common.environment('cuda'))
        p=json.loads((PANEL/'protocol.json').read_text())
        for job in p['jobs']:
            if job['architecture']!='A':continue
            check();row=p['parents'][job['parent']];ds=job['dataset'];seed=row['train_seed']
            path=PANEL/job['id']/(ds+'_T_eager')/'initial_complete.pt'
            state=torch.load(path,map_location='cpu',weights_only=False)
            parent=torch.load(row['readout'],map_location='cpu',weights_only=False);cfg=parent['binding']['config']
            assert state['updates']==0 and not state['optimizer']['state']
            assert s.state_digest(state['model'])==s.state_digest(parent['model'])
            z=np.load(row['data']);types=z['types'];pairs=z['schema'];ee=z['edges'];ee=ee if ee.shape[0]==2 else ee.T
            case=w.common.make_case(types,ee,tuple(map(tuple,pairs)))
            q=np.array(list(itertools.combinations(range(len(types)),2))).T
            lookup={tuple(r):j for j,r in enumerate(pairs)}
            f=np.array([lookup[tuple(sorted(types[e]))] for e in q.T])
            refset={tuple(sorted(e)) for e in ee.T};ref=np.array([tuple(e) in refset for e in q.T],float)
            prior=np.bincount(f,weights=ref,minlength=len(pairs))/np.bincount(f,minlength=len(pairs))
            keymap={tuple(e):j for j,e in enumerate(q.T)}
            frozen=json.loads((OLD/(ds+'_frozen_target.json')).read_text())
            m=w.StructuralDenoiser(case,cfg,'A').cuda();m.eval()
            oc={k:v for k,v in cfg['optimizer'].items() if k!='type'};oc['lr']=1e-5
            o=torch.optim.AdamW(m.parameters(),**oc);ema=s.ParameterEMA(m,.99);rng=np.random.default_rng()
            def restore():
                s.restore_payload(state,m,o,ema,state['binding'],lambda x:s.rng_restore(x,rng,'cuda'))
                o.zero_grad(set_to_none=True);m.eval()
                boundary=s.TrainingBoundary(m,o,ema,rng,'cuda',0)
                assert s.state_digest(boundary.checkpoint(state['binding']))==s.state_digest(state)
                return boundary
            boundary=restore()
            inputs={}
            for split,root,offset in [('update',202611130953,920000),('independent',202611133211,930000)]:
                inputs[split]=[w.common.train_input(case,1000,1,seed,offset+j,
                    np.random.default_rng(np.random.SeedSequence([root,seed,j])),force_t=t) for j,t in enumerate(TIMES)]
            hashes={split:[s.input_hash([i]) for i in ins] for split,ins in inputs.items()}
            history=json.loads((ROOT/'results/increment_update_check_20261113'/(job['id']+'.json')).read_text())
            assert hashes['update']==[r['input_hash'] for r in history['baseline']]
            assert not (set(hashes['update'])&set(hashes['independent']))
            def forward(inp):
                nonlocal calls
                check();logits,prob=w.runtime.predict(m,types,inp['edges'],inp['edge_families'],inp['query'],inp['f'],inp['t']/1000,details=True);calls+=1
                w.common.check_prob_device(prob,inp['f'],case,m.prior)
                return logits,prob
            def expand(prob,inp,requires_grad=False):
                ix=np.array([keymap[tuple(sorted(e))] for e in inp['query'].T])
                assert len(set(ix))==len(ix)
                np.testing.assert_array_equal(f[ix],inp['f']);np.testing.assert_array_equal(ref[ix],inp['labels'])
                pp=prior[f].copy();pp[ix]=prob.detach().cpu().double().numpy()
                miss=np.ones(len(q.T),bool);miss[ix]=False
                assert np.all((pp[miss]==0)|(pp[miss]==1))
                return torch.tensor(pp,dtype=torch.float64,requires_grad=requires_grad),ix
            def evaluate(tag):
                before=s.state_digest(boundary.checkpoint(state['binding']));rows=[];dump={}
                with torch.no_grad():
                    for split,ins in inputs.items():
                        for inp in ins:
                            logits,prob=forward(inp);tp,ix=expand(prob,inp)
                            vals,_=risks(tp,types,q,f,frozen)
                            pm=probability_metrics(tp.numpy(),types,q,f,ref,len(pairs))
                            terms=s.supported_binary_ce(logits,torch.as_tensor(inp['labels'],device='cuda'),m.prior[torch.as_tensor(inp['f'],device='cuda')]).cpu().numpy()
                            pp=prob.cpu().double().numpy();lab=inp['labels'];fam=inp['f'];relations=[]
                            for r in range(len(pairs)):
                                pos=(fam==r)&(lab==1);neg=(fam==r)&(lab==0)
                                relations.append(dict(relation=r,positive_n=int(pos.sum()),negative_n=int(neg.sum()),
                                    positive_CE=float(terms[pos].mean()) if pos.any() else None,negative_CE=float(terms[neg].mean()) if neg.any() else None,
                                    missed_mass=float((1-pp[pos]).sum()),false_mass=float(pp[neg].sum())))
                            rows.append(dict(split=split,t=inp['t'],input_hash=s.input_hash([inp]),
                                loss={k:float(v) for k,v in vals.items()},structure=pm,CE=float(terms.sum()/case['space'].total),
                                positive_CE=float(terms[lab==1].mean()),negative_CE=float(terms[lab==0].mean()),relations=relations))
                            dump[f'{split}_t{inp["t"]}']=tp.numpy()
                assert before==s.state_digest(boundary.checkpoint(state['binding']))
                np.savez_compressed(OUT/(job['id']+'_'+tag+'_probabilities.npz'),**dump,types=types,query=q,family=f,reference=ref)
                return rows
            result=dict(job=job['id'],dataset=ds,seed=seed,input_hashes=hashes,baseline=evaluate('baseline'),arms={},comparisons=[])
            initial={n:v.detach().cpu().clone() for n,v in m.named_parameters()};grads={};vectors={}
            for arm in ['bias','risk']:
                boundary=restore();gradlogit=[];pres=[];tick=time.time()
                with boundary.step(1):
                    before_rng=s.state_digest(s.rng_capture(rng,'cuda'))
                    for inp in inputs['update']:
                        logits,prob=forward(inp);tp,ix=expand(prob,inp,True)
                        vals,_=risks(tp,types,q,f,frozen)
                        derivative=torch.autograd.grad(vals[arm]/len(TIMES),tp)[0]
                        assert torch.isfinite(derivative).all()
                        # Exact chain rule at current p; no logit derivative
                        # substituted for dL/dp and no Monte Carlo surrogate.
                        prob.backward(derivative[ix].to(device=prob.device,dtype=prob.dtype));backs+=1
                        gradlogit.append(derivative.detach().numpy()*tp.detach().numpy()*(1-tp.detach().numpy()))
                        pres.append({k:float(v.detach()) for k,v in vals.items()})
                    assert before_rng==s.state_digest(s.rng_capture(rng,'cuda'))
                    grads[arm]={n:v.grad.detach().cpu().clone() if v.grad is not None else torch.zeros_like(v,device='cpu') for n,v in m.named_parameters()}
                    assert all(torch.isfinite(v).all() for v in grads[arm].values())
                    norm=float(torch.nn.utils.clip_grad_norm_(list(m.parameters()),1,error_if_nonfinite=True))
                    boundary.mark_optimizer_started();o.step();ema.update(m,1);ema.at_update_boundary=True
                    assert all(torch.isfinite(v).all() for v in m.parameters())
                vectors[arm]={n:v.detach().cpu()-initial[n] for n,v in m.named_parameters()}
                saved=boundary.checkpoint(state['binding']);assert saved['updates']==1 and ema.updates==1
                assert s.state_digest(saved['rng'])==s.state_digest(state['rng'])
                cp=OUT/(job['id']+'_'+arm+'_temporary.pt');torch.save(saved,cp)
                loaded=torch.load(cp,map_location='cpu',weights_only=False)
                assert s.state_digest(loaded)==s.state_digest(saved)
                s.restore_payload(loaded,m,o,ema,state['binding'],lambda x:s.rng_restore(x,rng,'cuda'));m.eval()
                assert s.state_digest(boundary.checkpoint(state['binding']))==s.state_digest(saved)
                result['arms'][arm]=dict(gradient_norm=norm,clip_scale=min(1,1/(norm+1e-6)),loss_before=pres,
                    update_seconds=time.time()-tick,validation=evaluate(arm),checkpoint_sha256=sha(cp),
                    serialized_restore_exact=True,RNG_preserved=True,CUDA_recompute_bitwise_certified=False)
                np.savez_compressed(OUT/(job['id']+'_'+arm+'_gradients.npz'),
                    **{'param_'+n:v.numpy() for n,v in grads[arm].items()},
                    **{'delta_'+n:v.numpy() for n,v in vectors[arm].items()},edge_logit_gradient=np.array(gradlogit))
                write(job['id']+'.json',result)
                write('progress.json',dict(job=job['id'],arm=arm,forwards=calls,backwards=backs,completed_models=len(done)))
            for kind,vs in [('gradient',grads),('update',vectors)]:
                for g in ['ALL','CN','joint_head','degree','backbone']:
                    names=[n for n in initial if g=='ALL' or group(n)==g]
                    result['comparisons'].append(dict(kind=kind,group=g,**cosine({n:vs['bias'][n] for n in names},{n:vs['risk'][n] for n in names})))
            write(job['id']+'.json',result);done.append(job['id'])
            del m,o,ema,state,parent,initial,grads,vectors,saved,loaded;torch.cuda.empty_cache()
        assert calls==192 and backs==48
        assert all(sha(path)==h for path,h in proto['bindings'].items())
        write('completion.json',dict(status='complete',models=done,forwards=calls,backwards=backs,temporary_updates=8,
            new_reverse_chains=0,peak_GPU_MiB=torch.cuda.max_memory_allocated()/2**20,original_assets_unchanged=True))
    except BaseException:
        write('failure.json',dict(traceback=traceback.format_exc(),models=done,forwards=calls,backwards=backs));raise

def launch():
    if (OUT/'launch.json').exists():raise RuntimeError('No duplicate launch')
    p=json.loads((OUT/'protocol.json').read_text())
    assert time.time()<p['science_stop_epoch']-750,'insufficient remaining full-panel time'
    assert all(sha(path)==h for path,h in p['bindings'].items())
    q=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    rows=[r.split(', ') for r in q.stdout.strip().splitlines()]
    gpu=next(r for r in rows if r[0]=='0');assert int(gpu[2])==0 and int(gpu[3])<=5
    start=time.time();env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0'
    write('launch.json',dict(allocation_start=start,GPU=0,uuid=gpu[1],resource_snapshot=q.stdout,
        stop_epoch=min(p['science_stop_epoch'],start+p['max_GPU_seconds'])))
    with (OUT/'worker.log').open('w') as log:
        child=subprocess.Popen([sys.executable,__file__,'worker'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write('pid.json',dict(pid=child.pid))
        try:child.wait(timeout=max(1,min(p['science_stop_epoch']-start,p['max_GPU_seconds'])-5))
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=2)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        finally:
            end=time.time();write('release.json',dict(exit_code=child.returncode,owned_process_exited=child.poll() is not None,
                allocation_start=start,end_epoch=end,GPU_hours=(end-start)/3600))
    if child.returncode:raise SystemExit(child.returncode)

if __name__=='__main__':
    {'prepare':prepare,'launch':launch,'worker':worker}[sys.argv[1]]()
