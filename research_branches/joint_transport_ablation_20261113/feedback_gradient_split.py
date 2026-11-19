"""Read-only VJPs of existing terminal-label feedback. Never optimizer.step()."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,time,csv,traceback,importlib.util,subprocess,signal
from pathlib import Path
from datetime import datetime
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/feedback_gradient_split_20261113T110439'
PANEL=ROOT/'results/joint_transport_ablation_20261113T011928'
OLD=ROOT/'results/robust_heat_update_check_20261113T095140'
START=datetime.fromisoformat('2026-11-13T11:04:39+08:00').timestamp()
STOP=START+720
def write(name,obj):
    OUT.mkdir(exist_ok=True,parents=True);(OUT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False))
def worker():
    spec=importlib.util.spec_from_file_location('split_support',ROOT/'research_branches/one_step_vs_terminal_plan_20261111/update_preflight.py')
    s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s);w,np,torch=s.w,s.np,s.torch
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    sources={};done=[];calls=backs=0
    def bind(path):sources[str(path)]=s.base.sha(path)
    def check():
        if time.time()>=STOP:raise TimeoutError('fixed science stop')
    def group(n):return 'CN' if n.startswith('common_neighbor.') else 'joint_head' if n.startswith('joint_head.') else 'degree' if n.startswith('branch.') else 'backbone'
    def relation_stats(v,f):
        v=np.asarray(v);constant=np.zeros_like(v)
        for r in np.unique(f):constant[f==r]=v[f==r].mean()
        ss=float(v@v)
        return dict(norm=float(np.linalg.norm(v)),sum=float(v.sum()),relation_constant_energy=float(constant@constant/ss) if ss else None)
    try:
        proto=json.loads((PANEL/'protocol.json').read_text());bind(PANEL/'protocol.json')
        for path,h in proto['bindings'].items():assert s.base.sha(path)==h,path
        scorepath=ROOT/'results/robust_heat_transport_20261113/training_rescore.csv';bind(scorepath);scores=list(csv.DictReader(scorepath.open()))
        write('environment.json',w.common.environment('cuda'))
        for job in proto['jobs']:
            if job['architecture']!='A' or job['train_seed'] not in [2026110411,2026110422]:continue
            check();ds=job['dataset'];seed=job['train_seed'];row=proto['parents'][job['parent']];folder=PANEL/job['id']/(ds+'_T_eager')
            path=folder/'initial_complete.pt';bind(path);state=torch.load(path,map_location='cpu',weights_only=False)
            bind(row['readout']);parent=torch.load(row['readout'],map_location='cpu',weights_only=False);cfg=parent['binding']['config']
            assert state['updates']==0 and not state['optimizer']['state'] and s.state_digest(state['model'])==s.state_digest(parent['model'])
            bind(row['data']);z=np.load(row['data']);ed=z['edges'];ed=ed if ed.shape[0]==2 else ed.T
            case=w.common.make_case(z['types'],ed,tuple(map(tuple,z['schema'])))
            m=w.StructuralDenoiser(case,cfg,'A').cuda();m.eval()
            oc={k:v for k,v in cfg['optimizer'].items() if k!='type'};oc['lr']=1e-5
            o=torch.optim.AdamW(m.parameters(),**oc);ema=s.ParameterEMA(m,.99);rng=np.random.default_rng()
            s.restore_payload(state,m,o,ema,state['binding'],lambda x:s.rng_restore(x,rng,'cuda'));m.eval()
            boundary=s.TrainingBoundary(m,o,ema,rng,'cuda',0);before=s.state_digest(boundary.checkpoint(state['binding']))
            assert before==s.state_digest(state)
            params=list(m.named_parameters());names=[n for n,p in params];pp=[p for n,p in params]
            split={k:{n:torch.zeros_like(p,device='cpu',dtype=torch.float64) for n,p in params} for k in ['label','prediction']}
            ids,fam,query=w.common.plan_for(case,2026111402,1).block(0)
            costs=[]
            for plan in job['terminal_plan']['1']:
                rr=[r for r in scores if r['dataset']==ds and int(r['train_seed'])==seed and r['arm']=='A' and int(r['update'])==1 and int(r['chain_seed'])==plan['seed']]
                assert len(rr)==1;costs.append(float(rr[0]['robust']))
            costs=np.array(costs);adv=(4*costs-costs.sum())/3
            arrays={'query':query,'family':fam,'ids':ids};per=[];summed={k:np.zeros(len(fam)) for k in ['label','prediction','total']};index=0
            for k,plan in enumerate(job['terminal_plan']['1']):
                path=folder/f'train_round1_seed{plan["seed"]}';bind(path/'identity.json');identity=json.loads((path/'identity.json').read_text())
                assert identity['model_hash']==s.state_digest(state['model']) and identity['updates']==0
                bind(path/'graph.npz');gr=np.load(path/'graph.npz');ee=gr['edges'];ee=ee if ee.shape[0]==2 else ee.T
                es={tuple(sorted(map(int,e))) for e in ee.T};yy=np.array([tuple(sorted(map(int,e))) in es for e in query.T],dtype=np.float64);assert yy.sum()==len(es)
                target=torch.as_tensor(yy,device='cuda');coef=.01*float(adv[k])/32
                for t in sorted(plan['times'],reverse=True):
                    check();xp=path/f'Xt_{t:04d}.npz';bind(xp);xx=np.load(xp);assert int(xx['t'])==t
                    ed,ff=w.common.state_edges(case,xx['xt']);logits,prob=w.runtime.predict(m,case['space'].types,ed,ff,query,fam,t/1000,details=True);calls+=1
                    w.common.check_prob_device(prob,fam,case,m.prior)
                    logits=logits.double();zz=logits[:,1]-logits[:,0];sig=torch.sigmoid(zz)
                    label_loss=coef*(target*zz).sum();prediction_loss=-coef*torch.nn.functional.softplus(zz).sum()
                    original=coef*torch.where(target.bool(),torch.log_softmax(logits,dim=-1)[:,1],torch.log_softmax(logits,dim=-1)[:,0]).sum()
                    err=float(abs(label_loss+prediction_loss-original));assert err<1e-8
                    for component,loss,retain in [('label',label_loss,True),('prediction',prediction_loss,False)]:
                        grad=torch.autograd.grad(loss,pp,retain_graph=retain,allow_unused=True);backs+=1
                        for (n,p),g in zip(params,grad):
                            if g is not None:
                                assert torch.isfinite(g).all();split[component][n]+=g.detach().cpu().double()
                        del grad
                    prob_cpu=sig.detach().cpu().numpy();dl=-coef*yy;dp=coef*prob_cpu;dt=dl+dp
                    for key,val in [('label',dl),('prediction',dp),('total',dt)]:summed[key]+=val
                    arrays[f'p_{index}']=prob_cpu;arrays[f'supported_p_{index}']=prob.detach().cpu().double().numpy();arrays[f'label_{index}']=yy;arrays[f'descent_{index}']=dt
                    per.append(dict(index=index,chain_seed=plan['seed'],t=t,advantage=float(adv[k]),loss=float(original),split_loss_error=err,label=relation_stats(dl,fam),prediction=relation_stats(dp,fam),total=relation_stats(dt,fam),by_relation=[dict(family=int(r),label=float(dl[fam==r].sum()),prediction=float(dp[fam==r].sum()),total=float(dt[fam==r].sum())) for r in np.unique(fam)]));index+=1
                    del logits,prob,zz,sig,label_loss,prediction_loss,original
            after=s.state_digest(boundary.checkpoint(state['binding']));assert before==after
            assert all(p.grad is None for p in pp)
            # Read old actual AdamW state, never apply any new update.
            oldpath=OLD/f'{job["id"]}_robust_temporary.pt';bind(oldpath);old=torch.load(oldpath,map_location='cpu',weights_only=False)
            oldreport=OLD/(job['id']+'.json');bind(oldreport);prev=json.loads(oldreport.read_text())['arms']['robust'];clip=prev['clip_scale'];beta=oc['betas'][0]
            pids=old['optimizer']['param_groups'][0]['params'];assert len(pids)==len(params)
            total={n:split['label'][n]+split['prediction'][n] for n in names}
            saved={};displacement={};matched=0
            for (n,p),pid in zip(params,pids):
                st=old['optimizer']['state'].get(pid)
                saved[n]=st['exp_avg'].double()/((1-beta)*clip) if st else torch.zeros_like(total[n]);matched+=st is not None
                displacement[n]=(old['model'][n]-state['model'][n]).double()
            groups=[]
            for g in ['ALL','backbone','CN','degree','joint_head','last4_attention','other_parameters']:
                def included(n):
                    limited=any(f'tf_layers.{i}.self_attn.' in n for i in range(4,8))
                    return g=='ALL' or (g=='last4_attention' and limited) or (g=='other_parameters' and not limited) or group(n)==g
                ns=[n for n in names if included(n)]
                dot=lambda a,b:sum(float(a[n].flatten()@b[n].flatten()) for n in ns)
                norm=lambda a:dot(a,a)**.5
                nl,npr,nt,no,nd=map(norm,[split['label'],split['prediction'],total,saved,displacement])
                error={n:total[n]-saved[n] for n in ns}
                groups.append(dict(group=g,parameters=sum(total[n].numel() for n in ns),label_norm=nl,prediction_norm=npr,total_norm=nt,
                    label_prediction_cosine=dot(split['label'],split['prediction'])/(nl*npr) if nl*npr else None,
                    residual_fraction=nt/(nl+npr) if nl+npr else None,saved_gradient_norm=no,
                    recompute_relative_error=norm(error)/no if no else None,
                    actual_old_displacement_norm=nd,old_descent_gradient_update_cosine=-dot(total,displacement)/(nt*nd) if nt*nd else None))
            torch.save(dict(label=split['label'],prediction=split['prediction'],total=total),OUT/(job['id']+'_gradients.pt'))
            np.savez_compressed(OUT/(job['id']+'_training_probabilities.npz'),**arrays)
            result=dict(dataset=ds,seed=seed,job=job['id'],costs=costs.tolist(),advantages=adv.tolist(),positions=per,logit_summed={k:relation_stats(v,fam) for k,v in summed.items()},logit_summed_by_relation=[dict(family=int(r),**{k:float(v[fam==r].sum()) for k,v in summed.items()}) for r in np.unique(fam)],groups=groups,old_optimizer_states=matched,full_state_unchanged=True,parameter_grads_absent=True,model_mode='eval',CUDA_bitwise_recompute_certified=False)
            write(job['id']+'.json',result);done.append(job['id']);write('progress.json',dict(conditions=done,forwards=calls,VJPs=backs))
            del m,o,ema,boundary,params,pp,state,parent,old,split,total,saved,displacement,arrays;torch.cuda.empty_cache()
        write('completion.json',dict(status='complete',conditions=done,forwards=calls,VJPs=backs,updates=0,new_graphs=0,end_epoch=time.time(),peak_memory_MiB=torch.cuda.max_memory_allocated()/2**20))
    except BaseException:write('failure.json',dict(error=traceback.format_exc(),conditions=done,forwards=calls,VJPs=backs,end_epoch=time.time()));raise
    finally:write('sources.json',sources)
def launch():
    write('protocol.json',dict(start_epoch=START,science_stop_epoch=STOP,delivery_epoch=START+900,max_GPU_hours=.15,device=0,conditions=['IMDB2026110411','PubMed2026110422'],forwards=64,VJPs=128,updates=0,new_graphs=0,rule='fixed saved robust reward; split a*y*z and -a*softplus(z); preserve full state; old AdamW state read only'))
    if time.time()>STOP-180:raise RuntimeError('Insufficient remaining time')
    start=time.time();env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0'
    with (OUT/'worker.log').open('w') as f:
        child=subprocess.Popen([sys.executable,__file__,'worker'],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        write('launch.json',dict(pid=child.pid,allocation_start=start,device=0,science_stop_epoch=STOP,delivery_epoch=START+900))
        try:child.wait(timeout=max(.1,min(STOP-start,535)))
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=3)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        finally:write('GPU_release.json',dict(exit_code=child.returncode,owned_process_exited=child.poll() is not None,allocation_start=start,end_epoch=time.time(),GPU_hours=(time.time()-start)/3600))
    if child.returncode:sys.exit(child.returncode)
if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='worker':worker()
    else:launch()
