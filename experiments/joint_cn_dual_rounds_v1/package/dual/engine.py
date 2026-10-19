import copy
import json
from pathlib import Path
import time
import uuid
import numpy as np
import torch
from hsgd.bw_preflight import common as c
from hsgd.runtime import reverse_step
from hsgd.kernel import MarginalKernel
from . import validation
from .data import degrees,digest
from .io import (Stopped,Control,atomic_json,log,configure,binding,optimizer,protect,
                 save_checkpoint,load_checkpoint)
from .metrics import graph,compare

def validate_fresh(out,fresh):
    out=protect(out)
    if not fresh:raise ValueError('first launch requires --fresh-start')
    if out.exists():raise ValueError('fresh output must not exist; use resume for THIS experiment')
    out.mkdir(parents=True)
    return out

def panel_binding(panel,cfg):
    import hashlib
    rows=[]
    for j,inp in enumerate(panel):
        h=hashlib.sha256()
        for key in ['t','edges','edge_families']:
            h.update(np.asarray(inp[key]).tobytes())
        rows.append(dict(t=inp['t'],noise_sha256=h.hexdigest(),plan_seed=[cfg['validation_seed'],j,21]))
    return rows

def run_train(case,cfg,out,device,fresh=False,resume=False,accept_gap=False):
    invocation_start=time.monotonic();cpu_start=time.process_time()
    if not accept_gap and not cfg['provenance']['formal_6000_entry_verified']:
        raise ValueError('old formal configuration receipt missing; explicitly acknowledge documented provenance gap')
    env=configure(device,False);base=binding(case,cfg,env)
    if resume:
        out=protect(out)
        if not out.is_dir():raise ValueError('resume output missing')
        bind=json.loads((out/'binding.json').read_text())
        if {k:v for k,v in bind.items() if k!='run_id'}!=base or not bind.get('run_id'):
            raise ValueError('resume belongs to another code/config/data/runtime')
        if (out/'STOP').exists():raise ValueError('remove this run STOP explicitly before resume')
    else:
        bind=dict(base,run_id=str(uuid.uuid4()))
        out=validate_fresh(out,fresh)
        atomic_json(out/'resolved_config.json',cfg);atomic_json(out/'binding.json',bind)
        atomic_json(out/'environment.json',env)
    c.seed_all(cfg['train_seed']);rng=np.random.default_rng(cfg['train_seed'])
    model=c.make_model(case,cfg,device);opt=optimizer(model,cfg)
    u=0
    if resume:u=load_checkpoint(out,model,opt,rng,device,bind)[0]['completed_updates']
    else:
        atomic_json(out/'dimensions.json',c.dimensions(model))
        save_checkpoint(out,model,opt,rng,device,0,bind)
    saved=c.rng_save(rng,device);panel=validation.panel(case,cfg);c.rng_restore(saved,rng,device)
    pb=panel_binding(panel,cfg)
    if resume and json.loads((out/'validation_panel.json').read_text())!=pb:raise ValueError('panel binding mismatch')
    atomic_json(out/'validation_panel.json',pb)
    started=time.monotonic();state='incomplete';reason=None;durations=[]
    sums=np.zeros((case['n_families'],2),np.int64);coverage=np.zeros(case['n_families'],np.int64)
    try:
        with Control(out) as ctl:
            val_u=set()
            if (out/'validation.jsonl').is_file():
                val_u={json.loads(line)['update'] for line in (out/'validation.jsonl').read_text().splitlines()}
            if u not in val_u and (u==0 or u%cfg['validation_every_updates']==0 or u==cfg['max_updates']):
                val=validation.run(model,case,cfg,device,panel,ctl.check)
                log(out/'validation.jsonl',dict(update=u,**val))
            while u<cfg['max_updates']:
                ctl.check();start=time.monotonic()
                inp=c.train_input(case,cfg['T'],cfg['rounds'],cfg['train_seed'],u,rng)
                result=c.update(model,opt,case,inp,cfg['T'],device,backend=cfg['backend'])
                # The core throws before this point on any invalid/nonfinite update.
                u+=1;elapsed=time.monotonic()-start;durations.append(elapsed)
                for label in (0,1):sums[:,label]+=np.bincount(inp['f'][inp['labels']==label],minlength=case['n_families'])
                coverage+=np.bincount(inp['f'],minlength=case['n_families'])
                log(out/'updates.jsonl',dict(update=u,learning_rate=opt.param_groups[0]['lr'],
                    total_update_seconds=elapsed,warmup_timing=len(durations)<=5,**result))
                atomic_json(out/'progress.json',dict(phase='training',completed_updates=u,loss=result['loss'],
                            max_updates=cfg['max_updates'],eight_hour_limit=False,backend=cfg['backend']))
                if u%100==0 or u==cfg['max_updates']:
                    log(out/'coverage.jsonl',dict(update=u,cumulative_queried_slots_by_family=coverage.tolist(),
                        cumulative_labels_by_family=sums.tolist(),scope='this invocation; visits, not unique coverage across updates'))
                if u%cfg['checkpoint_every_updates']==0:
                    save_checkpoint(out,model,opt,rng,device,u,bind)
                if u%cfg['validation_every_updates']==0 or u==cfg['max_updates']:
                    val=validation.run(model,case,cfg,device,panel,ctl.check)
                    log(out/'validation.jsonl',dict(update=u,**val))
            state='completed'
    except (Stopped,KeyboardInterrupt) as exc:
        reason=str(exc);state='stopped'
        # A full successful update is the only committed training unit.
        save_checkpoint(out,model,opt,rng,device,u,bind,kind='stop')
    except BaseException as exc:
        state='failed';reason=repr(exc)
        # Do not save a potentially partially updated model. Latest remains last durable valid checkpoint.
        raise
    finally:
        stable=durations[5:] if len(durations)>5 else []
        summary=dict(status=state,reason=reason,completed_updates=u,
            seconds_this_invocation=time.monotonic()-started,stable_updates=len(stable),
            total_worker_seconds_this_invocation=time.monotonic()-invocation_start,
            startup_seconds=started-invocation_start,cpu_seconds_this_invocation=time.process_time()-cpu_start,
            stable_seconds_median=float(np.median(stable)) if stable else None,
            stable_seconds_p90=float(np.quantile(stable,.9)) if stable else None,
            **c.memory(device),ordinary_mode_not_bitwise_reproducible=True)
        atomic_json(out/'training_summary.json',summary)
        log(out/'invocations.jsonl',summary)
    return state=='completed'

def sample(case,cfg,run,out,update,device,resume=False,engineering=False,max_steps=None):
    if update not in cfg['evaluation_updates']:raise ValueError('not a frozen evaluation checkpoint')
    env=configure(device,False);base=binding(case,cfg,env);run=Path(run)
    bind=json.loads((run/'binding.json').read_text())
    if {k:v for k,v in bind.items() if k!='run_id'}!=base or not bind.get('run_id'):raise ValueError('source training binding mismatch')
    out=protect(out)
    if resume:
        if (out/'STOP').exists():raise ValueError('remove sampling STOP before resume')
    elif out.exists():raise ValueError('sampling output must be new')
    else:out.mkdir(parents=True)
    c.seed_all(cfg['sample_seed']);rng=np.random.default_rng(cfg['sample_seed'])
    model=c.make_model(case,cfg,device);model.eval()
    ck,receipt=load_checkpoint(run,model,None,None,device,bind,update)
    sb=dict(binding=bind,checkpoint_sha256=receipt['sha256'],update=update,seed=cfg['sample_seed'],
            role_guidance=False,postprocessing=False)
    if resume:
        last=c.trusted_load(out/'state.pt')
        if last['binding']!=sb:raise ValueError('sample restore binding mismatch')
        state=last['state'];t=last['remaining'];c.rng_restore(last['rng'],rng,device)
    else:
        pi=np.repeat(case['prior'],case['space'].counts)
        state=(rng.random(case['space'].total)<pi).astype(np.uint8);t=cfg['T']
        atomic_json(out/'binding.json',sb)
    kernel=MarginalKernel(case['prior'],c.schedule(cfg['T'],cfg['schedule_offset']))
    def commit():
        c.torch_save(out/'state.pt',dict(binding=sb,state=state,remaining=t,rng=c.rng_save(rng,device)))
    commit();started=time.monotonic();status='incomplete';reason=None;steps=0
    try:
        with Control(out) as ctl:
            def check():
                ctl.check()
                if (run/'STOP').exists():raise Stopped('source pipeline STOP requested')
            while t>0:
                check()
                with torch.no_grad():new,report=reverse_step(model,case,state,kernel,t,cfg['rounds'],cfg['sample_seed'],rng,device,
                        backend=cfg['backend'],check_time=check)
                state=new;t-=1;steps+=1;commit()
                log(out/'steps.jsonl',report)
                atomic_json(out/'progress.json',dict(completed_steps=cfg['T']-t,remaining=t,seed=cfg['sample_seed']))
                if engineering and max_steps is not None and steps>=max_steps:raise Stopped('engineering step cap')
            # Recoverable final export; atomic state contains all committed work.
            edges,fam=c.state_edges(case,state)
            tmp=out/'graph.tmp.npz';np.savez_compressed(tmp,types=case['space'].types,edges=edges.T,relations=fam+1,
                schema=np.asarray(case['space'].pairs));tmp.replace(out/'graph.npz')
            check();metrics=graph(case,edges,cfg['data']['type_names'],check)
            ref=graph(case,case['clean'],cfg['data']['type_names'],check)
            atomic_json(out/'evaluation.json',dict(graph=metrics,reference=ref,comparison=compare(metrics,ref),
                        graph_sha256=c.sha(out/'graph.npz'),binding=sb,single_graph_exploratory=True))
            status='completed'
    except (Stopped,KeyboardInterrupt) as exc:
        reason=str(exc);status='stopped';commit()
    except BaseException as exc:
        reason=repr(exc);status='failed';raise
    finally:
        atomic_json(out/'summary.json',dict(status=status,reason=reason,remaining=t,completed_steps=cfg['T']-t,
            seconds_this_invocation=time.monotonic()-started,graph_sha256=c.sha(out/'graph.npz') if (out/'graph.npz').exists() else None,
            binding=sb,**c.memory(device)))
    return status=='completed'

def preflight(case,cfg,out,device):
    """Disposable strict REAL forward-noise update and short save/restore gate.

    This is NOT an independent legacy-vs-upstream proof (CN legacy aliases upstream).
    Formal model is created afresh in the subsequent separate process.
    """
    out=protect(out)
    if out.exists():raise ValueError('gate output exists')
    out.mkdir(parents=True);env=configure(device,True);c.seed_all(cfg['train_seed'])
    rng=np.random.default_rng(cfg['train_seed']);model=c.make_model(case,cfg,device);opt=optimizer(model,cfg)
    inp=c.train_input(case,cfg['T'],cfg['rounds'],cfg['train_seed'],0,rng,force_t=min(20,cfg['T']))
    first=c.update(model,opt,case,inp,cfg['T'],device,backend=cfg['backend'])
    snap=dict(model=c.cpu_tree(model.state_dict()),optimizer=c.cpu_tree(opt.state_dict()),rng=c.rng_save(rng,device))
    c.torch_save(out/'gate.pt',snap);snap=c.trusted_load(out/'gate.pt')
    inp2=c.train_input(case,cfg['T'],cfg['rounds'],cfg['train_seed'],1,rng,force_t=max(1,cfg['T']//2))
    c.update(model,opt,case,inp2,cfg['T'],device,backend=cfg['backend'])
    target=c.cpu_tree(model.state_dict());target_opt=c.cpu_tree(opt.state_dict());target_rng=c.rng_save(rng,device)
    model.load_state_dict(snap['model']);opt.load_state_dict(snap['optimizer']);c.rng_restore(snap['rng'],rng,device)
    inp3=c.train_input(case,cfg['T'],cfg['rounds'],cfg['train_seed'],1,rng,force_t=max(1,cfg['T']//2))
    c.update(model,opt,case,inp3,cfg['T'],device,backend=cfg['backend'])
    err=c.tree_error(target,c.cpu_tree(model.state_dict()));oe=c.tree_error(target_opt,c.cpu_tree(opt.state_dict()))
    if err>0 or oe>0:raise ValueError('strict save/restore update differs')
    # RNG is serialized/restored, including data generator; compare serialized state numerically.
    if digest(rng.bit_generator.state)!=digest(target_rng['generator']):raise ValueError('generator restore mismatch')
    atomic_json(out/'gate.json',dict(status='passed',binding=binding(case,cfg,env),checkpoint_sha256=c.sha(out/'gate.pt'),
        engine_updates=2,parameter_error=err,optimizer_error=oe,
        first_update=first,scope='disposable strict real-input gate; no formal updates counted'))

def verify_completed_sample(case,cfg,run,out,update):
    run,out=Path(run),Path(out)
    summary=json.loads((out/'summary.json').read_text());evaluation=json.loads((out/'evaluation.json').read_text())
    source=json.loads((run/'milestones.json').read_text())[str(update)]
    bind=dict(binding=json.loads((run/'binding.json').read_text()),checkpoint_sha256=source['sha256'],
              update=update,seed=cfg['sample_seed'],role_guidance=False,postprocessing=False)
    if c.sha(run/source['path'])!=source['sha256']:raise ValueError('source checkpoint changed')
    if summary['status']!='completed' or summary['remaining']!=0 or summary['completed_steps']!=cfg['T'] or summary['binding']!=bind:
        raise ValueError('completed sample counters/binding mismatch')
    sha=c.sha(out/'graph.npz')
    if summary['graph_sha256']!=sha or evaluation['graph_sha256']!=sha or evaluation['binding']!=bind or 'comparison' not in evaluation:
        raise ValueError('completed evaluation/graph evidence mismatch')
    ck=c.trusted_load(out/'state.pt')
    if ck['binding']!=bind or ck['remaining']!=0:raise ValueError('completed state evidence mismatch')
    edges,fam=c.state_edges(case,ck['state'])
    with np.load(out/'graph.npz',allow_pickle=False) as z:
        if not (np.array_equal(z['types'],case['space'].types) and np.array_equal(z['edges'],edges.T)
                and np.array_equal(z['relations'],fam+1) and np.array_equal(z['schema'],case['space'].pairs)):
            raise ValueError('graph differs from committed final state')
    return True
