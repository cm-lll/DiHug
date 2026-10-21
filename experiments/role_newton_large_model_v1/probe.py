"""Complete20-block frozen-Xt step, cold and fresh repeated inference. CPU only."""
import argparse,resource,signal,traceback,copy
from large_common import *

class Measurement:
    def __init__(self,m,folder,check_fn):
        self.m=m;self.folder=folder;self.check=check_fn;self.times={};self.reads=[];self.arrays={};self.hooks=[];self.values=[];self.phase='cold';self.blocks=[]
    def add(self,key,dt):self.times[key]=self.times.get(key,0.)+dt
    def module_timer(self,module,key):
        stack=[]
        def before(m,args):stack.append(time.perf_counter())
        def after(m,args,out):self.add(key,time.perf_counter()-stack.pop())
        self.hooks.extend([module.register_forward_pre_hook(before),module.register_forward_hook(after)])
    def install(self):
        self.original=(cap.allocation_marginals,cap.degree_logpmf,cap.degrees)
        def law(*args,**kwargs):
            tick=time.perf_counter();v=self.original[1](*args,**kwargs);self.add('degree_law_cache_construction_seconds',time.perf_counter()-tick);self.times['new_degree_law_tables']=self.times.get('new_degree_law_tables',0)+1;self.check();return v
        def degree(*args,**kwargs):
            tick=time.perf_counter();v=self.original[2](*args,**kwargs);self.add('current_relation_degrees_seconds',time.perf_counter()-tick);return v
        def allocation(L,counts):
            self.check();tick=time.perf_counter();P,LP,r=solver.solve(L,counts,self.check)
            if not r.get('new_contract_pass'):
                np.savez_compressed(self.folder/(self.phase+'_failed_matrix.npz'),L=L,counts=counts);save(self.folder/(self.phase+'_failed_matrix.json'),r)
                raise solver.ReadoutFailure(str(r))
            self.add('Newton_solver_seconds',r['seconds']);self.add('Newton_support_seconds',r['support_seconds'])
            vt=time.perf_counter();ind=g.verify(P,LP,L,counts,r);self.add('independent_read_validation_seconds',time.perf_counter()-vt)
            typ=len(self.reads);self.reads.append(dict(type=typ,**r,**ind));self.arrays.update({f'type{typ}_L':L.copy(),f'type{typ}_counts':counts.copy(),f'type{typ}_P':P.copy(),f'type{typ}_logP':LP.copy()})
            self.add('Newton_plus_validation_seconds',time.perf_counter()-tick);return P,r
        cap.allocation_marginals=allocation;cap.degree_logpmf=law;cap.degrees=degree
        for module,key in [(self.m.backbone,'transformer_backbone_seconds'),(self.m.branch.degree_encoder,'degree_encoder_seconds'),(self.m.branch.degree_out,'degree_projection_seconds'),(self.m.branch.role_encoder,'role_encoder_seconds'),(self.m.branch.value,'role_value_seconds'),(self.m.role_fusion,'role_fusion_projection_seconds')]:self.module_timer(module,key)
        for module in self.m.joint_head:self.module_timer(module,'joint_readout_seconds')
        def vh(mod,args,value):self.values.append(value.detach().cpu().numpy().copy())
        def bh(mod,args,output):
            assert len(self.reads)==len(self.values)==self.m.n_types
            self.diagnostic={k:float(v) for k,v in output[1].items() if k.endswith('_seconds')}
            context=output[1]['role_context'].detach().cpu().numpy()
            types=self.case['space'].types
            for r,v in zip(self.reads,self.values):
                assert np.isfinite(v).all();P=self.arrays[f"type{r['type']}_P"];rows=np.asarray(r['tail_row_mass']);M=float(abs(v).max())
                r['context_tail_bound']=float(rows.max(initial=0)*M);r['role_value_max']=M;assert r['context_tail_bound']<=1e-250
                f32=P.astype(np.float32).astype(float);k=P.shape[1];gamma=k*2**-24/(1-k*2**-24)
                conversion=(abs(f32-P)@abs(v.astype(float))).max(1);dot=(gamma*(abs(f32)@abs(v.astype(float)))).max(1)+k*np.finfo(np.float32).tiny
                observed=abs(context[types==r['type']]-(P@v.astype(float))).max(1)
                assert np.all(observed<=conversion+dot+1e-12)
                r.update(context_FP32_conversion_bound=float(conversion.max()),context_FP32_dot_bound=float(dot.max()),context_observed_error=float(observed.max()))
        self.hooks.extend([self.m.branch.value.register_forward_hook(vh),self.m.branch.register_forward_hook(bh)])
        cn=self.m.common_neighbor;self.old_contextual=cn.contextual_nodes;self.old_aggregate=cn.aggregate
        def contextual(*args,**kwargs):
            tick=time.perf_counter();v=self.old_contextual(*args,**kwargs);self.add('CN_existing_Xt_node_encoding_seconds',time.perf_counter()-tick);return v
        def aggregate(*args,**kwargs):
            tick=time.perf_counter();v=self.old_aggregate(*args,**kwargs);self.add('CN_full_aggregation_seconds',time.perf_counter()-tick);return v
        cn.contextual_nodes=contextual;cn.aggregate=aggregate
    def reset(self,phase):self.phase=phase;self.times={};self.reads=[];self.arrays={};self.values=[];self.blocks=[];self.diagnostic={}
    def close(self):
        cap.allocation_marginals,cap.degree_logpmf,cap.degrees=self.original
        self.m.common_neighbor.contextual_nodes=self.old_contextual;self.m.common_neighbor.aggregate=self.old_aggregate
        for h in self.hooks:h.remove()

def run(name,t):
    start=time.perf_counter();cpu=time.process_time();folder=OUT/f'{name}_t{t}';folder.mkdir();status='failed';reason=None;stage='binding';monitor=None;rows=[]
    def stop(s,f):(folder/'STOP').touch()
    for s in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(s,stop)
    def gate():
        check()
        if (folder/'STOP').exists():raise InterruptedError('item STOP')
        if time.perf_counter()-start>=410:raise TimeoutError('item410s cooperative reserve10s')
    try:
        gate();f=frozen();meta=next(x for x in f['inputs'] if x['dataset']==name and x['t']==t);assert meta['resource_gate_pass']
        resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
        case,target,cfg,inv,path=dataset(name)
        with np.load(OUT/meta['path']) as z:state=z['state'].copy()
        baseline_peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        baseline_rss=float(next(line.split()[1] for line in Path('/proc/self/status').read_text().splitlines() if line.startswith('VmRSS:')))/1024
        stage='model_construction';tick=time.perf_counter();model=make_model(case,cfg);construction=time.perf_counter()-tick
        bind=read(OUT/f'{name}_binding.json');assert state_digest(model)==bind['random_weight_state_sha256'];model.set_target(target)
        assert not torch.cuda.is_initialized();monitor=Measurement(model,folder,gate);monitor.case=case;monitor.install()
        kernel=c.base.MarginalKernel(case['prior'],c.base.c.schedule(1000));old_state=state.copy();model_hash=state_digest(model)
        outputs=[];random_before=None
        for phase in ('cold','repeat'):
            gate();stage=phase+'_full_step';monitor.reset(phase)
            if phase=='cold':model.branch.tables.clear()
            cache_before=len(model.branch.tables);rng=np.random.default_rng([DRAW_SEED,0 if name=='IMDB' else 1,t]);rng_before=copy.deepcopy(rng.bit_generator.state)
            if random_before is None:random_before=rng_before
            else:assert random_before==rng_before
            def event(kind,**kwargs):
                if kind=='reverse_block':
                    monitor.blocks.append(dict(**kwargs,elapsed=time.perf_counter()-tick));print(name,t,phase,'block',kwargs['block_completed'],'/20',flush=True)
                    save(folder/'progress.json',dict(stage=stage,completed_blocks=kwargs['block_completed'],elapsed_seconds=time.perf_counter()-start))
            tick=time.perf_counter()
            with torch.no_grad():new,info=c.base.reverse_step(model,case,state,kernel,t,20,meta['draw_seed'],rng,'cpu',event=event,check_time=gate,trace=True)
            full_seconds=time.perf_counter()-tick;step_peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
            assert np.array_equal(state,old_state) and len(monitor.reads)==4
            gate();stage=phase+'_audit';at=time.perf_counter();trace=info['trace'];expected=case['active'].repeat(case['space'].counts).astype(np.uint8)
            np.testing.assert_array_equal(trace['coverage'],expected);assert info['blocks']==20 and info['covered_slots']==sum(meta['query']['query_sizes'])
            fam=np.repeat(np.arange(case['n_families']),case['space'].counts);pt=time.perf_counter();q=kernel.reverse(trace['p0'],state,fam,t-1,t);posterior_audit_seconds=time.perf_counter()-pt
            np.testing.assert_array_equal(q,trace['q']);pt=time.perf_counter();replayed_commit=(trace['uniform']<q).astype(np.uint8);sampling_replay_seconds=time.perf_counter()-pt
            np.testing.assert_array_equal(new,replayed_commit)
            if phase=='repeat':
                for k in ('p0','q','coverage','state'):np.testing.assert_array_equal(trace[k],outputs[0][k])
            outputs.append({k:v.copy() for k,v in trace.items() if k!='uniform'})
            edges,families=c.base.c.state_edges(case,new);ff,loc=case['space'].encode(edges)
            assert np.array_equal(ff,families) and len(np.unique(case['space'].offsets[ff]+loc))==len(families) and np.all(edges[0]<edges[1])
            assert np.array_equal(model.prior.numpy(),case['prior']) and state_digest(model)==model_hash
            audit_seconds=time.perf_counter()-at;gate();stage=phase+'_IO';it=time.perf_counter()
            p0=trace['p0'].astype(np.float32);np.testing.assert_array_equal(p0.astype(float),trace['p0'])
            np.savez_compressed(folder/(phase+'_output.npz'),p0=p0,q=trace['q'],state=new,coverage=trace['coverage'],edges=edges,families=families)
            np.savez_compressed(folder/(phase+'_readouts.npz'),**monitor.arrays)
            save(folder/(phase+'_reads.json'),monitor.reads);io_seconds=time.perf_counter()-it
            timing=info['timings'];role_attention=monitor.diagnostic['role_attention_seconds']
            residual=max(0.,role_attention-monitor.times.get('degree_law_cache_construction_seconds',0)-monitor.times.get('Newton_plus_validation_seconds',0)-monitor.times.get('role_encoder_seconds',0)-monitor.times.get('role_value_seconds',0))
            row=dict(phase=phase,completed_full_step=True,network_freshly_executed=True,full_step_seconds=full_seconds,audit_seconds=audit_seconds,IO_seconds=io_seconds,
                runtime_timings=timing,module_timings=monitor.times,condition_diagnostic=monitor.diagnostic,
                likelihood_lookup_assembly_and_role_loop_residual_seconds=residual,blocks=20,covered_slots=info['covered_slots'],coverage_sha256=hashlib.sha256(trace['coverage'].tobytes()).hexdigest(),
                input_state_sha256=hashlib.sha256(state.tobytes()).hexdigest(),output_state_sha256=hashlib.sha256(new.tobytes()).hexdigest(),
                uniform_sha256=hashlib.sha256(trace['uniform'].tobytes()).hexdigest(),rng_before=rng_before,rng_after=rng.bit_generator.state,
                cache_tables_before=cache_before,cache_tables_after=len(model.branch.tables),cache_bytes=sum(v.nbytes for v in model.branch.tables.values()),
                cache_reuse='repeat only degree-law tables for identical t; no graph/hidden/P/logit/q reuse',read_count=len(monitor.reads),read_statuses=[r['status'] for r in monitor.reads],
                peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,output_sha256=sha(folder/(phase+'_output.npz')),readouts_sha256=sha(folder/(phase+'_readouts.npz')),
                peak_rss_after_step_mib=step_peak,independent_full_posterior_audit_seconds=posterior_audit_seconds,independent_sampling_commit_replay_seconds=sampling_replay_seconds,
                trace_overhead='original trace=True computes diagnostic posterior once additionally per block and saves p0/q; same policy both paths')
            rows.append(row);save(folder/'paths.json',rows);gate()
        status='completed';stage='completed'
    except BaseException as exc:reason=repr(exc);traceback.print_exc()
    finally:
        if monitor is not None:
            save(folder/'last_read_receipts.json',monitor.reads);monitor.close()
        save(folder/'receipt.json',dict(status=status,reason=reason,failure_stage=stage,dataset=name,t=t,completed_paths=len(rows),wall_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu,
            baseline_current_rss_mib=locals().get('baseline_rss'),baseline_prior_peak_rss_mib=locals().get('baseline_peak'),peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            model_construction_seconds=locals().get('construction'),random_weights=True,quality_evaluation=False,training=0,GPU=0))
    return int(status!='completed')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset',choices=('IMDB','PubMed'));p.add_argument('t',type=int,choices=(20,500,1000));a=p.parse_args();sys.exit(run(a.dataset,a.t))
