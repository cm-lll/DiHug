"""One-step CE versus frozen heterogeneous micro/macro reference; no chains.

CPU analytic derivatives are bridged to the unchanged CUDA probability head.
Fresh feedback Adam state is shared by both arms, not historical training Adam.
"""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[k]='1'
import sys,json,time,hashlib,signal,subprocess,traceback,importlib.util,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/hetero_micro_macro_model_check_20261116'
PANEL=ROOT/'results/joint_transport_ablation_20261113T011928'
OLD=ROOT/'results/analytic_joint_risk_20261113'
PREV=ROOT/'results/analytic_joint_model_check_20261113'
FEATURE=ROOT/'results/hetero_micro_macro_cpu_20261116_v2'
CAL=ROOT/'results/hetero_micro_macro_calibration_20261116'
TIMES=[1,75,250,550,800,1000]
START=1794784292.0  # 2026-11-15 23:11:32 UTC; never reset on launch.
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def write(n,x): (OUT/n).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False))
def prepare():
    OUT.mkdir(exist_ok=False)
    old=json.loads((PREV/'protocol.json').read_text());bindings=dict(old['bindings'])
    # Old experiment script is evidence; hash all live dependencies and frozen assets.
    for p in [Path(__file__),Path(__file__).with_name('objective.py'),CAL/'frozen_scales.json',
              FEATURE/'IMDB_frozen.json',FEATURE/'PubMed_frozen.json',PANEL/'protocol.json']:
        bindings[str(p)]=sha(p)
    for job in old['jobs']: bindings[str(PREV/(job+'.json'))]=sha(PREV/(job+'.json'))
    for p,h in bindings.items(): assert sha(p)==h,p
    assert time.time()<START+1200-600,'Insufficient time for full bounded panel'
    write('protocol.json',dict(start_epoch=START,science_stop_epoch=START+1080,delivery_epoch=START+1200,
        max_GPU_seconds=900,jobs=old['jobs'],bindings=bindings,arms=['CE','micro_macro'],
        forwards=82,backwards=10,scientific_updates=8,recovery_update_executions=2,new_graphs=0,
        update_t=550,readout_times=TIMES,readout='raw only; existing viewed independent split, not new holdout',
        optimizer='same sealed EMA3000 parent, fresh feedback AdamW lr1e-5, original clip1, EMA.99',
        loss='original CE coefficient1 + frozen lambda mean_type_block(error/(4sigma_squared))',
        limitations=['No historical Adam continuation','Repeated CUDA updates not certified bitwise',
        'Conditional clean prediction proxies, not generation quality','No per-result threshold or weight changes']))
def worker():
    visible=os.environ['CUDA_VISIBLE_DEVICES']
    sys.path.insert(0,str(ROOT/'research_branches/analytic_joint_risk_20261113'))
    from core import risks,probability_metrics
    os.environ['CUDA_VISIBLE_DEVICES']=visible
    from objective import make_plan,descriptors,distribution_errors
    spec=importlib.util.spec_from_file_location('support',ROOT/'research_branches/one_step_vs_terminal_plan_20261111/update_preflight.py')
    s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
    w,np,torch=s.w,s.np,s.torch
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    proto=json.loads((OUT/'protocol.json').read_text());launch=json.loads((OUT/'launch.json').read_text())
    stop=min(proto['science_stop_epoch'],launch['allocation_start']+895)
    scale=json.loads((CAL/'frozen_scales.json').read_text());calls=backs=0;done=[]
    def check():
        if time.time()>=stop:raise TimeoutError('Bounded science/GPU limit')
    def compare(a,b):
        dif=[]
        def visit(x,y,path):
            if torch.is_tensor(x):
                assert x.shape==y.shape and x.dtype==y.dtype
                dif.append((float((x.double()-y.double()).abs().max()) if x.numel() else 0.,path))
            elif isinstance(x,dict):
                assert x.keys()==y.keys()
                for k in x:visit(x[k],y[k],path+'/'+str(k))
            elif isinstance(x,(list,tuple)):
                assert len(x)==len(y)
                for i in range(len(x)):visit(x[i],y[i],path+'/'+str(i))
            else: assert s.state_digest(x)==s.state_digest(y),path
        visit(a,b,'');return dict(exact=s.state_digest(a)==s.state_digest(b),max_absolute_difference=max(dif,default=(0,'')))
    try:
        for path,h in proto['bindings'].items():assert sha(path)==h,path
        write('environment.json',w.common.environment('cuda'))
        p=json.loads((PANEL/'protocol.json').read_text())
        for job in p['jobs']:
            if job['architecture']!='A':continue
            check();row=p['parents'][job['parent']];ds=job['dataset'];seed=row['train_seed']
            state=torch.load(PANEL/job['id']/(ds+'_T_eager')/'initial_complete.pt',map_location='cpu',weights_only=False)
            parent=torch.load(row['readout'],map_location='cpu',weights_only=False);cfg=parent['binding']['config']
            assert state['updates']==0 and not state['optimizer']['state']
            assert s.state_digest(state['model'])==s.state_digest(parent['model'])
            z=np.load(row['data']);types=z['types'];pairs=z['schema'];ee=z['edges'];ee=ee if ee.shape[0]==2 else ee.T
            case=w.common.make_case(types,ee,tuple(map(tuple,pairs)))
            q=np.array(list(itertools.combinations(range(len(types)),2))).T
            lookup={tuple(r):j for j,r in enumerate(pairs)};f=np.array([lookup[tuple(sorted(types[e]))] for e in q.T])
            refset={tuple(sorted(e)) for e in ee.T};ref=np.array([tuple(e) in refset for e in q.T],float)
            prior=np.bincount(f,weights=ref,minlength=len(pairs))/np.bincount(f,minlength=len(pairs))
            keymap={tuple(e):j for j,e in enumerate(q.T)};plan=make_plan(pairs,len(np.unique(types)))
            frozen_degree=json.loads((OLD/(ds+'_frozen_target.json')).read_text())
            frozen=json.loads((FEATURE/(ds+'_frozen.json')).read_text())
            frozen={int(t):{g:dict(scale=torch.tensor(b['scale'],dtype=torch.float64),omega=torch.tensor(b['omega'],dtype=torch.float64),
                target=torch.tensor(np.array(b['target_real'])+1j*np.array(b['target_imag']),dtype=torch.complex128)) for g,b in groups.items()} for t,groups in frozen.items()}
            m=w.StructuralDenoiser(case,cfg,'A').cuda();m.eval()
            oc={k:v for k,v in cfg['optimizer'].items() if k!='type'};oc['lr']=1e-5
            o=torch.optim.AdamW(m.parameters(),**oc);ema=s.ParameterEMA(m,.99);rng=np.random.default_rng()
            def restore(st):
                s.restore_payload(st,m,o,ema,st['binding'],lambda x:s.rng_restore(x,rng,'cuda'))
                o.zero_grad(set_to_none=True);m.eval()
                b=s.TrainingBoundary(m,o,ema,rng,'cuda',st['updates'])
                assert s.state_digest(b.checkpoint(st['binding']))==s.state_digest(st)
                return b
            boundary=restore(state);inputs={}
            for split,root,offset in [('update',202611130953,920000),('independent',202611133211,930000)]:
                inputs[split]=[w.common.train_input(case,1000,1,seed,offset+j,np.random.default_rng(np.random.SeedSequence([root,seed,j])),force_t=t) for j,t in enumerate(TIMES)]
            hashes={k:[s.input_hash([i]) for i in ins] for k,ins in inputs.items()}
            assert hashes==json.loads((PREV/(job['id']+'.json')).read_text())['input_hashes']
            def forward(inp):
                nonlocal calls
                check();lo,pr=w.runtime.predict(m,types,inp['edges'],inp['edge_families'],inp['query'],inp['f'],inp['t']/1000,details=True);calls+=1
                w.common.check_prob_device(pr,inp['f'],case,m.prior);return lo,pr
            def expand(pr,inp,grad=False):
                ix=np.array([keymap[tuple(sorted(e))] for e in inp['query'].T]);assert len(set(ix))==len(ix)
                np.testing.assert_array_equal(f[ix],inp['f']);np.testing.assert_array_equal(ref[ix],inp['labels'])
                pp=prior[f].copy();pp[ix]=pr.detach().cpu().double().numpy();missing=np.ones(len(pp),bool);missing[ix]=False
                assert np.all((pp[missing]==0)|(pp[missing]==1))
                return torch.tensor(pp,dtype=torch.float64,requires_grad=grad),ix
            def macro(tp):
                P=tp.new_zeros((len(pairs),len(types),len(types)));P[f,q[0],q[1]]=tp;P[f,q[1],q[0]]=tp
                errors=[x for x in distribution_errors(descriptors(P,types,plan),types,frozen) if x['block']!='degree']
                _,parts=risks(tp,types,q,f,frozen_degree)
                errors += [dict(type=x['type'],block='degree',error=x['risk']) for x in parts]
                val=scale['lambda_macro']*torch.stack([x['error']/(4*scale['sigma_squared'][ds][str(x['type'])][x['block']]) for x in errors]).mean()
                return val,errors
            def terms(lo,inp):return s.supported_binary_ce(lo,torch.as_tensor(inp['labels'],device='cuda'),m.prior[torch.as_tensor(inp['f'],device='cuda')])
            def evaluate(tag):
                before=s.state_digest(boundary.checkpoint(state['binding']));rows=[];dump={}
                with torch.no_grad():
                    for inp in inputs['independent']:
                        lo,pr=forward(inp);tp,ix=expand(pr,inp);ma,errors=macro(tp)
                        ce=terms(lo,inp).cpu().numpy();lab=inp['labels'];fam=inp['f'];pp=pr.cpu().double().numpy();rels=[]
                        for r in range(len(pairs)):
                            pos=(fam==r)&(lab==1);neg=(fam==r)&(lab==0)
                            rels.append(dict(relation=r,positive_n=int(pos.sum()),negative_n=int(neg.sum()),positive_CE=float(ce[pos].mean()) if pos.any() else None,
                                negative_CE=float(ce[neg].mean()) if neg.any() else None,missed_mass=float((1-pp[pos]).sum()),false_mass=float(pp[neg].sum()),expected_edges=float(tp[f==r].sum())))
                        rows.append(dict(t=inp['t'],input_hash=s.input_hash([inp]),CE=float(ce.sum()/case['space'].total),positive_CE=float(ce[lab==1].mean()),negative_CE=float(ce[lab==0].mean()),
                            macro=float(ma),errors=[dict(type=x['type'],block=x['block'],error=float(x['error'])) for x in errors],relations=rels,
                            structure=probability_metrics(tp.numpy(),types,q,f,ref,len(pairs))))
                        dump['t'+str(inp['t'])]=tp.numpy()
                assert before==s.state_digest(boundary.checkpoint(state['binding']))
                np.savez_compressed(OUT/(job['id']+'_'+tag+'_probabilities.npz'),**dump,types=types,query=q,family=f,reference=ref)
                return rows
            def update(arm,step):
                nonlocal backs
                tick=time.time();inp=inputs['update'][3]
                with boundary.step(step):
                    lo,pr=forward(inp);ce=terms(lo,inp).sum()/case['space'].total
                    if arm=='micro_macro':
                        tp,ix=expand(pr,inp,True);ma,_=macro(tp);dp=torch.autograd.grad(ma,tp)[0]
                        assert torch.isfinite(dp).all()
                        torch.autograd.backward([ce,pr],[torch.ones_like(ce),dp[ix].to(pr)])
                    else:ma=torch.tensor(0.);ce.backward()
                    backs+=1
                    grad={n:v.grad.detach().cpu().clone() if v.grad is not None else torch.zeros_like(v,device='cpu') for n,v in m.named_parameters()}
                    assert all(torch.isfinite(v).all() for v in grad.values())
                    norm=float(torch.nn.utils.clip_grad_norm_(m.parameters(),1,error_if_nonfinite=True))
                    boundary.mark_optimizer_started();o.step();ema.update(m,step);ema.at_update_boundary=True
                    assert all(torch.isfinite(v).all() for v in m.parameters())
                saved=boundary.checkpoint(state['binding']);assert saved['updates']==step and ema.updates==step
                assert s.state_digest(saved['rng'])==s.state_digest(state['rng'])
                return saved,grad,dict(CE_before=float(ce.detach()),macro_before=float(ma.detach()),gradient_norm=norm,seconds=time.time()-tick)
            result=dict(job=job['id'],dataset=ds,seed=seed,input_hashes=hashes,baseline=evaluate('baseline'),arms={})
            initial={n:v.detach().cpu().clone() for n,v in m.named_parameters()};vectors={};grads={}
            for arm in ['CE','micro_macro']:
                boundary=restore(state);saved,grad,record=update(arm,1);grads[arm]=grad
                vectors[arm]={n:v.detach().cpu()-initial[n] for n,v in m.named_parameters()}
                cp=OUT/(job['id']+'_'+arm+'_temporary.pt');torch.save(saved,cp)
                loaded=torch.load(cp,map_location='cpu',weights_only=False);assert s.state_digest(saved)==s.state_digest(loaded)
                boundary=restore(loaded);record.update(serialized_restore_exact=True,validation=evaluate(arm),checkpoint_sha256=sha(cp),CUDA_recompute_bitwise_certified=False)
                result['arms'][arm]=record;write(job['id']+'.json',result)
                if not done and arm=='micro_macro':
                    boundary=restore(saved);a,_,_=update(arm,2)
                    boundary=restore(torch.load(cp,map_location='cpu',weights_only=False));b,_,_=update(arm,2)
                    result['recovery']={k:compare(a[k],b[k]) for k in ['model','optimizer','ema','rng','updates']}
                    result['recovery']['note']='Exact serialized restore required; no changed CUDA recomputation tolerance, differences reported, not certified bitwise.'
                    boundary=restore(saved)
            result['directions']=[]
            for kind,vs in [('gradient',grads),('update',vectors)]:
                for group in ['ALL','CN','joint_head','degree','backbone']:
                    def grp(n):return 'CN' if n.startswith('common_neighbor.') else 'joint_head' if n.startswith('joint_head.') else 'degree' if n.startswith('branch.') else 'backbone'
                    names=[n for n in initial if group=='ALL' or grp(n)==group]
                    aa=sum(float(vs['CE'][n].double().square().sum()) for n in names);bb=sum(float(vs['micro_macro'][n].double().square().sum()) for n in names)
                    dot=sum(float((vs['CE'][n].double()*vs['micro_macro'][n].double()).sum()) for n in names)
                    result['directions'].append(dict(kind=kind,group=group,CE_norm=aa**.5,candidate_norm=bb**.5,cosine=dot/(aa*bb)**.5 if aa*bb else None))
            write(job['id']+'.json',result);done.append(job['id'])
            write('progress.json',dict(done=done,forwards=calls,backwards=backs))
            del m,o,ema,state,parent,initial,grads,vectors,saved,loaded;torch.cuda.empty_cache()
        assert calls==82 and backs==10
        for path,h in proto['bindings'].items():assert sha(path)==h,path
        write('completion.json',dict(status='complete',models=done,forwards=calls,backwards=backs,temporary_scientific_updates=8,recovery_updates=2,new_graphs=0,peak_GPU_MiB=torch.cuda.max_memory_allocated()/2**20,original_assets_unchanged=True))
    except BaseException:
        write('failure.json',dict(traceback=traceback.format_exc(),done=done,forwards=calls,backwards=backs));raise
def launch():
    if (OUT/'launch.json').exists():raise RuntimeError('No repeated launch')
    p=json.loads((OUT/'protocol.json').read_text());assert time.time()<p['science_stop_epoch']-480,'Insufficient remaining time'
    resource=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    gpu=resource.splitlines()[0].split(', ');assert gpu[0]=='0' and int(gpu[2])==0 and int(gpu[3])<=5
    start=time.time();env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0'
    write('launch.json',dict(allocation_start=start,GPU=0,uuid=gpu[1],snapshot=resource,stop_epoch=min(p['science_stop_epoch'],start+900)))
    with (OUT/'worker.log').open('w') as log:
        child=subprocess.Popen([sys.executable,__file__,'worker'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write('pid.json',dict(pid=child.pid,watchdog_pid=os.getpid()))
        try:child.wait(timeout=max(1,min(p['science_stop_epoch']-start,895)))
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=2)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        finally:
            end=time.time();write('release.json',dict(exit_code=child.returncode,owned_process_exited=child.poll() is not None,allocation_start=start,end_epoch=end,GPU_hours=(end-start)/3600))
    if child.returncode:raise SystemExit(child.returncode)
if __name__=='__main__':{'prepare':prepare,'launch':launch,'worker':worker}[sys.argv[1]]()
