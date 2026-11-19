"""Eight isolated updates from four saved boundaries; no new training trajectory."""
import os,sys,time,json,signal,traceback,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(os.environ['OBJECTIVE_DIAG_OUT'])
OLD=ROOT/'results/reproducible_objective_20261108T135622/attempt2'
spec=importlib.util.spec_from_file_location('bounded_old',ROOT/'results/output_boundary_replay_20261108T081654/worker.py')
w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
sys.path.insert(0,str(ROOT/'research_branches/reproducible_aggregation_20261108'))
from adapter import execution_policy,VERSION
g,fr,torch,np=w.g,w.fr,w.torch,w.np
P=w.read(OUT/'protocol.json');start=time.time();stopped=False
status=dict(status='loading',start_epoch=start,updates=0,forwards=0,gradient_probes=0,completed=[])
def halt(*a):
    global stopped
    stopped=True
for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,halt)
def check():
    if stopped or time.time()>=min(P['science_stop_epoch'],start+870):raise TimeoutError('scientific/GPU limit')
def save(name,x):fr.save(OUT/name,x)
def flat(values):return torch.cat([x.detach().cpu().flatten() for x in values])
def dot(a,b):return float(torch.dot(a.double(),b.double()))
def norm(a):return float(torch.linalg.vector_norm(a.double()))
def cos(a,b):
    den=norm(a)*norm(b)
    return dot(a,b)/den if den else None
def main():
    fr.setup();fr.authenticate_sources()
    assert torch.cuda.is_available() and torch.cuda.device_count()==1 and '2080 Ti' in torch.cuda.get_device_name(0)
    save('environment.json',dict(**g.common.environment('cuda'),execution_policy=VERSION))
    artifacts=w.read(OLD/'audit.json')['local_artifacts'];results=[];rows=[];bindings=[]
    tasks=w.read(ROOT/'results/output_boundary_replay_20261108T081654/tasks.json')
    for task in tasks:
        ds,cfg=task['dataset'],task['row']['config'];check()
        path=ROOT/g.SCI.relative_to(g.ROOT)/'package'/cfg['datasets'][ds]['file']
        assert w.sha(path)==cfg['datasets'][ds]['sha256']
        with np.load(path,allow_pickle=False) as z:case=g.common.make_case(z['types'],z['edges'].T,tuple(map(tuple,z['schema'])))
        fixed=torch.load(OLD/f'{ds}_fixed_inputs.pt',map_location='cpu')['inputs']
        fixed={t:inp for panel,t,seed,inp in fixed if panel=='new_diagnostic'}
        history=[json.loads(x) for x in (w.OLD/'run'/task['task']/'training.jsonl').read_text().splitlines()]
        m=g.StructuralDenoiser(case,cfg,'A').cuda();o=torch.optim.AdamW(m.parameters(),**{k:v for k,v in cfg['optimizer'].items() if k!='type'})
        e=g.ParameterEMA(m);r=np.random.default_rng(1);ps=w.make_objective(case,cfg,ds,'PS')
        named=list(m.named_parameters());params=[p for n,p in named]
        spans=[];cursor=0
        for n,p in named:spans.append((n,w.category(n),cursor,cursor+p.numel()));cursor+=p.numel()
        def grads(loss,retain):
            values=torch.autograd.grad(loss,params,retain_graph=retain,allow_unused=True)
            status['gradient_probes']+=1
            return flat([torch.zeros_like(p) if v is None else v for p,v in zip(params,values)])
        b=None;parent=None;binding=None
        def restore():
            nonlocal b
            work=w.cpu_copy(parent);g.restore_payload(work,m,o,e,work['binding'],lambda x:g.rng_restore(x,r,'cuda'))
            m.train();o.zero_grad(set_to_none=True);b=g.TrainingBoundary(m,o,e,r,'cuda',parent['updates'])
            live=b.checkpoint(binding)
            for k in ['model','optimizer','ema','rng','updates']:assert g.digest(live[k])==g.digest(parent[k]),'restore '+k
        def metrics(inp,logits,p):
            y=torch.as_tensor(inp['labels'],device='cuda');terms=w.supported_binary_ce(logits,y,m.prior[torch.as_tensor(inp['f'],device='cuda')])
            ce=terms.detach().cpu().numpy();prob=p.detach().cpu().numpy();labels=inp['labels'];relations=[]
            for f in range(case['n_families']):
                ix=inp['f']==f;pos=ix&(labels==1);neg=ix&(labels==0)
                pred=np.zeros(len(case['space'].types));target=np.zeros_like(pred)
                for ends in inp['query']:np.add.at(pred,ends[ix],prob[ix]);np.add.at(target,ends[ix],labels[ix])
                nodes=np.isin(case['space'].types,case['space'].pairs[f])
                relations.append(dict(family=f,positive_count=int(pos.sum()),negative_count=int(neg.sum()),positive_CE=float(ce[pos].mean()) if pos.any() else None,
                    negative_CE=float(ce[neg].mean()) if neg.any() else None,expected_edges=float(prob[ix].sum()),true_edges=int(pos.sum()),
                    missed_positive_mass=float((1-prob[pos]).sum()),false_positive_mass=float(prob[neg].sum()),expected_relation_degree_MAE=float(abs(pred[nodes]-target[nodes]).mean()) if nodes.any() else None))
            result=dict(CE=float(ce.sum()/case['space'].total),positive_CE=float(ce[labels==1].mean()),negative_CE=float(ce[labels==0].mean()),expected_edges=float(prob.sum()),relations=relations)
            assert np.isfinite(ce).all() and np.isfinite(prob).all()
            return result,terms
        def evaluate(condition,arm,inp,label,need_grad=False,train_mode=False):
            check();before=g.digest(b.checkpoint(binding));gg=None
            with b.validation():
                m.train(train_mode);token=(condition,arm,label)
                ctx=g.FrozenTypedContext(m,case['space'].types,inp['edges'],inp['edge_families'],token)
                try:
                    with torch.set_grad_enabled(need_grad):
                        graph,logits,p=ctx.predict(inp['query'],inp['f'],inp['t']/1000,token,details=True);status['forwards']+=1
                        g.common.check_prob_device(p,inp['f'],case,m.prior)
                        row,terms=metrics(inp,logits,p)
                        y=torch.as_tensor(inp['labels'],device='cuda')
                        if need_grad:
                            if train_mode:
                                aux,_=ps(p,inp['query'],inp['f'],inp,logits=logits)
                                gg=dict(train_positive=grads(terms[y==1].sum()/case['space'].total,True),train_negative=grads(terms[y==0].sum()/case['space'].total,True),auxiliary=grads(aux,False))
                                row['PS']=float(aux.detach())
                            else:gg=dict(eval_positive=grads(terms[y==1].mean(),True),eval_negative=grads(terms[y==0].mean(),False))
                        row.update(condition=condition,arm=arm,dataset=ds,parent_branch=condition.split('_')[-1],t=inp['t'],panel=label,input_sha256=w.input_hash([inp]),completed=b.completed)
                        rows.append(row);np.savez_compressed(OUT/f'{condition}_{arm}_{label}.npz',probability=p.detach().cpu().numpy(),CE=terms.detach().cpu().numpy())
                finally:ctx.close()
            assert g.digest(b.checkpoint(binding))==before,'probe state contamination'
            save('evaluations.json',rows)
            return row,gg
        class Objective:
            def __init__(self,kind):self.kind=kind;self.last={}
            def weights(self,q,f,inp,*,device,dtype):return torch.ones(len(f),device=device,dtype=dtype)
            def __call__(self,p,q,f,inp,*,logits):return (p.sum()*0,p.new_empty(0)) if self.kind=='CE' else ps(p,q,f,inp,logits=logits)
        with execution_policy(True):
            for parent_arm in ['CE','PS']:
                check();condition=ds+'_'+parent_arm;cp=OLD/f'{ds}_{parent_arm}_diagnostic_state.pt'
                oldkey=str(cp.relative_to(ROOT));assert w.sha(cp)==artifacts[oldkey]
                envelope=torch.load(cp,map_location='cpu');assert envelope['diagnostic_only'] and envelope['local_objective']==parent_arm and envelope['execution_policy']==VERSION
                parent=envelope['state'];binding=parent['binding'];assert parent['updates']==task['start']+32 and binding['config']==cfg
                assert ps.objective_binding()==binding['output_objective']['target']
                restore();before=flat(params);u=b.completed+1
                # Draw once for the read-only probes, then restore the full boundary.
                inp=g.one_input(case,cfg,u,r);ih=w.input_hash([inp]);assert ih==history[u-1]['input_sha256']
                assert g.digest(g.rng_capture(r,'cuda'))==history[u-1]['full_rng_before'];restore()
                bindings.append(dict(condition=condition,parent=str(cp),sha256=w.sha(cp),update=u,t=inp['t'],input_sha256=ih,seed=task['seed']))
                save('assets.json',bindings)
                train_before,train_grad=evaluate(condition,'NONE',inp,'current_training',True,True)
                base={};eval_grad=None
                for t in [1,500,1000]:
                    base[t],gg=evaluate(condition,'NONE',fixed[t],f'fixed_{t}',t==500)
                    if gg is not None:eval_grad=gg
                outcomes={};deltas={};actual_grad={};states={}
                for kind in ['CE','PS']:
                    restore();obj=Objective(kind);captured={}
                    def capture(*a):
                        captured['gradient']=flat([torch.zeros_like(p) if p.grad is None else p.grad for p in params])
                        captured['gradient_none']=[p.grad is None for p in params]
                    hook=o.register_step_pre_hook(capture)
                    tick=time.perf_counter()
                    try:
                        with b.step(u):
                            actual=g.one_input(case,cfg,u,r);assert w.input_hash([actual])==ih
                            assert g.digest(g.rng_capture(r,'cuda'))==history[u-1]['full_rng_before']
                            rec=w.update(m,o,case,actual,1000,'cuda',e,u-1,transaction=b,objective=obj);status['updates']+=1;status['forwards']+=1
                            assert g.digest(g.rng_capture(r,'cuda'))==history[u-1]['full_rng_after']
                    finally:hook.remove()
                    elapsed=time.perf_counter()-tick
                    deltas[kind]=flat(params)-before;actual_grad[kind]=captured['gradient']
                    predicted=train_grad['train_positive']+train_grad['train_negative']+(train_grad['auxiliary'] if kind=='PS' else 0)
                    error=actual_grad[kind]-predicted
                    assert torch.allclose(actual_grad[kind],predicted,atol=2e-6,rtol=1e-5),'gradient sum mismatch'
                    train_after,_=evaluate(condition,kind,inp,'current_training',False,True)
                    responses={}
                    for t in [1,500,1000]:responses[t],_=evaluate(condition,kind,fixed[t],f'fixed_{t}')
                    outcomes[kind]=dict(training_record=rec,training_after=train_after,responses=responses,seconds=elapsed,gradient_none=captured['gradient_none'],gradient_sum_max_abs=float(error.abs().max()),gradient_sum_L2=norm(error))
                    states[kind]=b.checkpoint(binding)
                    fr.atomic_checkpoint(OUT/f'{condition}_{kind}_temporary.pt',dict(diagnostic_only=True,parent_sha256=w.sha(cp),intervention=kind,state=states[kind]))
                dd=deltas['PS']-deltas['CE'];gce=train_grad['train_positive']+train_grad['train_negative'];ga=train_grad['auxiliary']
                response={}
                for label,metric in [('eval_positive','positive_CE'),('eval_negative','negative_CE')]:
                    response[label]={}
                    for kind in ['CE','PS']:
                        actual=outcomes[kind]['responses'][500][metric]-base[500][metric];linear=dot(eval_grad[label],deltas[kind])
                        response[label][kind]=dict(actual_change=actual,linear_prediction=linear,remainder=actual-linear)
                    actual=outcomes['PS']['responses'][500][metric]-outcomes['CE']['responses'][500][metric]
                    response[label]['PS_minus_CE']=dict(actual_change=actual,linear_prediction=dot(eval_grad[label],dd),remainder=actual-dot(eval_grad[label],dd))
                groups={}
                for group in ['all','CN_exclusive','joint_head','degree_branch','shared_backbone_other']:
                    ids=[(lo,hi) for n,c,lo,hi in spans if group=='all' or c==group]
                    cat=lambda v:torch.cat([v[lo:hi] for lo,hi in ids])
                    groups[group]=dict(parameters=sum(hi-lo for lo,hi in ids),CE_gradient_L2=norm(cat(gce)),aux_gradient_L2=norm(cat(ga)),CE_aux_cosine=cos(cat(gce),cat(ga)),
                        CE_displacement_L2=norm(cat(deltas['CE'])),PS_displacement_L2=norm(cat(deltas['PS'])),displacement_cosine=cos(cat(deltas['CE']),cat(deltas['PS'])),difference_L2=norm(cat(dd)),
                        eval_positive_difference_linear=dot(cat(eval_grad['eval_positive']),cat(dd)),eval_negative_difference_linear=dot(cat(eval_grad['eval_negative']),cat(dd)),
                        eval_positive_aux_gradient_dot=dot(cat(eval_grad['eval_positive']),cat(ga)),eval_negative_aux_gradient_dot=dot(cat(eval_grad['eval_negative']),cat(ga)))
                # Exact Adam formulas are nonlinear in the final parameter step.
                # Moment deltas have separate known algebra; retain rounding residuals.
                ids=states['CE']['optimizer']['param_groups'][0]['params'];beta1,beta2=cfg['optimizer']['betas'];moment={}
                ceopt=states['CE']['optimizer']['state'];psopt=states['PS']['optimizer']['state']
                assert ceopt.keys()==psopt.keys() and outcomes['CE']['gradient_none']==outcomes['PS']['gradient_none']
                active=[(k,lo,hi) for k,(n,c,lo,hi) in zip(ids,spans) if k in ceopt and not outcomes['CE']['gradient_none'][ids.index(k)]]
                for key in ['exp_avg','exp_avg_sq']:
                    actual=flat([psopt[k][key]-ceopt[k][key] for k,lo,hi in active])
                    cg=torch.cat([actual_grad['CE'][lo:hi] for k,lo,hi in active]).double()
                    pg=torch.cat([actual_grad['PS'][lo:hi] for k,lo,hi in active]).double()
                    expected=(1-beta1)*(pg-cg) if key=='exp_avg' else (1-beta2)*(pg.square()-cg.square())
                    moment[key]=dict(actual_difference_L2=norm(actual),formula_difference_L2=norm(expected),rounding_residual_L2=norm(actual.double()-expected))
                assert all(torch.equal(ceopt[k]['step'],psopt[k]['step']) for k in ceopt)
                for k,lo,hi in active:
                    old_step=int(parent['optimizer']['state'][k]['step']) if k in parent['optimizer']['state'] else 0
                    assert int(ceopt[k]['step'])==old_step+1
                result=dict(condition=condition,dataset=ds,parent_branch=parent_arm,update=u,training_t=inp['t'],train_before=train_before,base=base,outcomes=outcomes,
                    gradient_geometry=groups,local_response=response,moment_differences=moment,current_CE_aux_cosine=cos(gce,ga),aux_over_CE_gradient_L2=norm(ga)/norm(gce),
                    learning_rate=cfg['optimizer']['lr'],weight_decay=cfg['optimizer']['weight_decay'],betas=cfg['optimizer']['betas'],eps=cfg['optimizer']['eps'])
                results.append(result);save('results.json',results)
                torch.save(dict(parameter_names=[n for n,p in named],spans=spans,training_gradients=train_grad,evaluation_gradients=eval_grad,actual_gradients=actual_grad,displacements=deltas),OUT/f'{condition}_local_tensors.pt')
                assert w.sha(cp)==artifacts[oldkey]
                status['completed'].append(condition);save('progress.json',status)
                del states,deltas,actual_grad,train_grad,eval_grad,envelope
        del m,o,e,b,parent,ps
        torch.cuda.empty_cache()
    fr.authenticate_sources();assert status['updates']==8 and status['forwards']==56 and status['gradient_probes']==20
    status['status']='complete'
if __name__=='__main__':
    try:main()
    except BaseException as exc:status.update(status='stopped_or_failed',error=repr(exc),traceback=traceback.format_exc())
    finally:
        status.update(end_epoch=time.time(),seconds=time.time()-start,peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20 if torch.cuda.is_initialized() else 0)
        save('receipt.json',status);print(json.dumps(status),flush=True)
    raise SystemExit(0 if status['status']=='complete' else 2)
