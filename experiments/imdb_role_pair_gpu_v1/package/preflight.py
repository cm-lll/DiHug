"""Disposable backend gate; never exports a formal initialization/checkpoint."""
import argparse,copy,json,os,statistics,time
from pathlib import Path
import numpy as np
import torch
import pair_core as p
from pair_model import create,paired_initial,Reader
from hsgd.computational import FrozenTypedContext
from role_condition.model import extract_target

ATOL=2e-6;RTOL=1e-5
def close(a,b):torch.testing.assert_close(a,b,atol=ATOL,rtol=RTOL)
class CostCounters:
    """No new per-layer synchronization; stream times are NOT kernel busy time."""
    def __init__(self,m):self.model=m;self.h=[];self.reset()
    def reset(self):self.cpu_law=0.;self.events=[];self.calls=[]
    def install(self):
        import capacity_readout as cap
        self.original=cap.degree_logpmf
        def law(*a,**kw):
            start=time.perf_counter();v=self.original(*a,**kw);self.cpu_law+=time.perf_counter()-start;return v
        cap.degree_logpmf=law
        stack=[]
        def before(module,args):
            e=torch.cuda.Event(enable_timing=True) if next(module.parameters()).is_cuda else None
            if e is not None:e.record()
            stack.append((time.perf_counter(),e))
        def after(module,args,value):
            cpu,e=stack.pop();end=torch.cuda.Event(enable_timing=True) if e is not None else None
            if end is not None:end.record();self.events.append((e,end))
            self.calls.append(time.perf_counter()-cpu)
        self.h=[self.model.backbone.register_forward_pre_hook(before),self.model.backbone.register_forward_hook(after)]
        return self
    def result(self):
        return {'CPU_degree_law_construction_seconds':self.cpu_law,'backbone_host_forward_seconds':sum(self.calls),
            'backbone_device_stream_seconds':sum(a.elapsed_time(b)/1000 for a,b in self.events),
            'backbone_calls':len(self.calls),'stream_interval_includes_dispatch_gaps_not_device_busy_time':True,'extra_per_layer_synchronization':False}
    def close(self):
        import capacity_readout as cap
        cap.degree_logpmf=self.original
        for h in self.h:h.remove()
def gate(out,device,cfg,check):
    p.setup(device,True);case=p.toy();target=extract_target(case)
    checks=[];records={};start=time.perf_counter()
    for arm in ('base','role'):
        print('START strict small numerical/update/resume gate',arm,flush=True)
        check();cpu=create(case,cfg,991,arm,'cpu');gpu=create(case,cfg,992,arm,device)
        # Exercise a NONZERO role contribution in an isolated gate fixture.
        # Formal fresh initialization keeps the original zero projection.
        with torch.no_grad():cpu.role_fusion.weight.fill_(.01)
        initial=p.cpu(cpu.state_dict());gpu.load_state_dict(initial,strict=True)
        target_for=target if arm=='role' else None;cpu.set_target(target_for);gpu.set_target(target_for)
        inputs=p.input_pair(case,cfg,993,0);diagnostics={};outputs=[]
        for model in (cpu,gpu):
            reader=Reader(check,verify_context=True).install(model)
            try:
                model.eval();inp=inputs[0]
                with torch.no_grad():
                    l,q=p.predict(model,case['space'].types,inp['edges'],inp['edge_families'],inp['query'],inp['f'],inp['t']/cfg['T'],details=True)
                    outputs.append((l.cpu(),q.cpu()))
                opt=p.optimizer(model,cfg);row,grads=p.update(model,opt,case,inputs,cfg,gradients=True)
                diagnostics['CPU' if model is cpu else 'target']=(row,grads,p.cpu(model.state_dict()),p.cpu(opt.state_dict()))
            finally:reader.close()
        close(outputs[0][0],outputs[1][0]);close(outputs[0][1],outputs[1][1])
        x,y=diagnostics['CPU'],diagnostics['target'];close(torch.tensor(x[0]['loss']),torch.tensor(y[0]['loss']))
        if x[1].keys()!=y[1].keys():raise ValueError('participating parameter sets differ')
        for k in x[1]:close(x[1][k],y[1][k])
        # Cross-backend Adam weights are descriptive, as in the existing gate:
        # tiny signed gradient differences near epsilon can amplify first steps.
        max_parameter=max(float((x[2][k]-y[2][k]).abs().max()) for k in x[2])
        gpu.load_state_dict(initial,strict=True);opt=p.optimizer(gpu,cfg);reader=Reader(check,verify_context=True).install(gpu)
        rng=np.random.default_rng(994)
        try:
            one,_=p.update(gpu,opt,case,inputs,cfg)
            if p.digest(gpu.state_dict())!=p.digest(y[2]):raise AssertionError('strict same-device first update not exact')
            payload={'binding':{'fixture':arm},'updates':1,'models':p.cpu(gpu.state_dict()),'optimizer':p.cpu(opt.state_dict()),'rng':p.rng_save(rng,device)}
            folder=out/('gate_'+arm);p.checkpoint(folder,payload)
            cont,_=p.update(gpu,opt,case,inputs,cfg);expected=p.cpu(gpu.state_dict());expected_opt=p.cpu(opt.state_dict())
            fresh=create(case,cfg,995,arm,device);fresh.set_target(target_for);fo=p.optimizer(fresh,cfg)
            ck=p.load_checkpoint(folder,{'fixture':arm});fresh.load_state_dict(ck['models'],strict=True);fo.load_state_dict(ck['optimizer']);p.rng_restore(ck['rng'],rng,device)
            if p.digest(fresh.state_dict())!=p.digest(ck['models']) or p.digest(fo.state_dict())!=p.digest(ck['optimizer']) or p.digest(p.rng_save(rng,device))!=p.digest(ck['rng']):raise AssertionError('loaded bytes/state differ')
            # Reader hook must follow the restored model, never the discarded one.
            reader.close();reader=Reader(check,verify_context=True).install(fresh)
            resumed,_=p.update(fresh,fo,case,inputs,cfg)
            for k,v in expected.items():close(v,fresh.state_dict()[k].cpu())
            if p.digest(fo.state_dict())!=p.digest(expected_opt):raise AssertionError('strict optimizer continuation differs')
            records[arm]={'logits_max_abs':float((outputs[0][0]-outputs[1][0]).abs().max()),'p0_max_abs':float((outputs[0][1]-outputs[1][1]).abs().max()),
                'gradient_max_abs':max(float((x[1][k]-y[1][k]).abs().max()) for k in x[1]),'cross_backend_Adam_parameter_max_abs_descriptive':max_parameter,
                'strict_repeat_exact':True,'strict_save_resume_pass':True,'parameter_changed':bool(one['parameter_changed']),'participating_gradients':len(x[1]),'CPU_GPU_updated_weights_required_identical':False}
            if not one['parameter_changed']:raise AssertionError('no optimizer update')
        finally:reader.close()
        checks.append(arm+'_forward_loss_gradients_strict_update_resume')
        print('END strict small gate',arm,'passed',flush=True)
    return {'checks':checks,'arms':records,'seconds':time.perf_counter()-start,'tolerance':{'atol':ATOL,'rtol':RTOL},'strict':True,
        'temporary_optimizer_updates':10,'CPU_oracle_updates':2,'target_device_updates':8,'same_process_fresh_object_resume':True}

def cost(out,device,cfg,check):
    p.setup(device,False);case,target=p.dataset(cfg);models=paired_initial(case,cfg,cfg['train_seeds'][0]);records={}
    for model in models:
        print('START complete IMDB cost',model.arm,flush=True)
        check();model.to(device);p.condition(model,target);opt=p.optimizer(model,cfg);reader=Reader(check,out/'failed_reads').install(model);counter=CostCounters(model).install()
        if device=='cuda':torch.cuda.reset_peak_memory_stats(0)
        try:
            updates=[];init_start=time.perf_counter()
            for u in range(5):
                check();counter.reset();input_tick=time.perf_counter();inp=p.input_pair(case,cfg,1991,u);row,_=p.update(model,opt,case,inp,cfg,profile=u==4)
                row['input_plus_update_seconds']=time.perf_counter()-input_tick
                row['cost_counters']=counter.result()
                row['kind']='warmup' if u==0 else ('detailed_diagnostic' if u==4 else 'steady_low_interference')
                updates.append(row);p.save(out/f"cost_{model.arm}_progress.json",{'phase':'temporary_updates','completed':u+1})
                print('temporary update',model.arm,u+1,row['kind'],row['input_plus_update_seconds'],flush=True)
            kernel=p.MarginalKernel(case['prior'],p.c.schedule(cfg['T']))
            inp=p.c.train_input(case,cfg['T'],cfg['rounds'],1992,0,np.random.default_rng(1993),force_t=500);state=p.state_from_edges(case,inp['edges'])
            steps=[]
            model.branch.tables.clear() # Explicit first t500 table construction.
            model.eval()
            with torch.no_grad():
                for j in range(2):
                    check();counter.reset();nr=len(reader.rows)
                    def event(name,**kw):
                        if name=='reverse_block':print('full step block',model.arm,j,kw['block_completed'],flush=True)
                    value,info=p.reverse_step(model,case,state,kernel,500,cfg['rounds'],1994,np.random.default_rng(1995),device,check_time=check,trace=False,event=event)
                    steps.append(dict(info,state_sha256=p.digest(value),cost_counters=counter.result(),CPU_Newton_seconds=sum(x['CPU_wall_seconds'] for x in reader.rows[nr:])))
                    p.save(out/f"cost_{model.arm}_progress.json",{'phase':'complete_twenty_block_steps','completed':j+1})
            if steps[0]['state_sha256']!=steps[1]['state_sha256']:
                # Ordinary mode may differ at thresholds; record, never silently
                # call it numerical non-equivalence or promise exact trajectories.
                repeat_identical=False
            else:repeat_identical=True
            vals=[r['input_plus_update_seconds'] for r in updates if r['kind']=='steady_low_interference']
            records[model.arm]={'updates':updates,'temporary_updates':5,'scientific_updates':0,'full_twenty_block_steps':steps,
                'repeat_state_identical':repeat_identical,'steady_update_median':float(np.median(vals)),'steady_update_P90':float(np.quantile(vals,.9)),
                'step_median_seconds':float(np.median([x['step_seconds'] for x in steps])),'role_reader_calls':len(reader.rows),
                'CPU_Newton_seconds':sum(x['CPU_wall_seconds'] for x in reader.rows),'reader_receipts':reader.rows,'memory':p.c.memory(device),
                'device_memory':p.environment(device),'dimensions':p.c.dimensions(model),'complete_validation_measured':False,'wall_seconds':time.perf_counter()-init_start}
            if model.arm=='base' and reader.rows:raise AssertionError('base consumed roles')
            p.save(out/f'cost_{model.arm}.json',records[model.arm])
        finally:counter.close();reader.close();model.to('cpu');del opt
    # Forecast uses measured complete steps; fixed full-cover validation costs
    # are conservatively estimated by five steps, explicitly not a measurement.
    compute=3*cfg['max_updates']*sum(x['steady_update_median'] for x in records.values())
    sampling=3*len(cfg['evaluation_updates'])*len(cfg['sampling_seeds'])*cfg['T']*sum(x['step_median_seconds'] for x in records.values())
    validation=3*(cfg['max_updates']//cfg['validation_every_updates']+1)*len(cfg['validation_times'])*sum(x['step_median_seconds'] for x in records.values())
    estimate=(compute+sampling+validation)*1.25+600
    return {'arms':records,'forecast':{'training_seconds':compute,'sampling_seconds_static_step_assumption':sampling,'validation_seconds_estimated_not_measured':validation,
        'total_seconds_with_25percent_and_600s_margin':estimate,'weights_random_or_temporary':True,'density_and_read_difficulty_may_change':True},'complete':True}

def main():
    a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True);a.add_argument('--device',choices=['cpu','cuda'],default='cuda');a.add_argument('--fixture-only',action='store_true');a.add_argument('--deadline',type=float,required=True);x=a.parse_args()
    out=x.output;cfg=p.config();status='failed';error=None;data={};start=time.time()
    def check():
        if (out/'STOP').exists():raise InterruptedError('STOP')
        if time.time()>x.deadline-10:raise TimeoutError('preflight deadline; reserve cleanup')
    try:
        check();manifest=p.verify_package();p.setup(x.device,True);p.acquire_guard(out,x.device);env=p.environment(x.device);p.save(out/'environment.json',env)
        if x.device=='cpu' and not x.fixture_only:raise ValueError('CPU gate is fixture-only; no GPU certification')
        free=__import__('shutil').disk_usage(out).free
        if free<cfg['minimum_free_disk_bytes']:raise OSError('persistent output free space below16GiB; do not overwrite existing files')
        data['numerical_gate']=gate(out,x.device,cfg,check);p.save(out/'numerical_gate.json',data['numerical_gate'])
        p.setup(x.device,False)
        if not x.fixture_only:data['cost']=cost(out,x.device,cfg,check)
        data.update(package_manifest_sha256=manifest,config_sha256=p.sha(p.ROOT/'config.json'),runtime_binding=p.bound_environment(p.environment(x.device)))
        status='passed_engineering_fixture' if x.fixture_only else 'passed_target_GPU'
    except BaseException as exc:error=repr(exc);raise
    finally:
        resources=p.c.memory('cpu')
        if x.device=='cuda' and torch.cuda.is_initialized():resources.update(p.c.memory('cuda'))
        p.save(out/'preflight.json',dict(status=status,error=error,checks=data.get('numerical_gate',{}).get('checks',[]),
            target_GPU_gate_passed=status=='passed_target_GPU',full_IMDB_cost_complete='cost' in data,wall_seconds=time.time()-start,
            scientific_updates=0,complete_T1000_chains=0,resources=resources,**data))
        print('PREFLIGHT',status,'scientific_updates=0','output='+str(out),flush=True)
if __name__=='__main__':main()
