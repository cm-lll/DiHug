"""Production dispatch and transactional frozen-Xt steps, v6.

No hidden representation cache; each context is owned by exactly one call or
one reverse time step. Legacy means the actual original model.forward.
"""
import time
import numpy as np
import torch
from hsgd.computational import FrozenTypedContext

BACKENDS = ("upstream", "legacy")


def backend_name(name):
    if name not in BACKENDS:
        raise ValueError("unknown graph backend: " + str(name))
    return name


def timed(device, times, key, fn):
    if times is None: return fn()
    if device == "cuda": torch.cuda.synchronize()
    start = time.perf_counter()
    cpu_start = time.process_time()
    first,last = None,None
    if device == "cuda":
        first,last=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        first.record()
    value = fn()
    if last is not None:last.record()
    if device == "cuda": torch.cuda.synchronize()
    times[key] = times.get(key, 0.) + time.perf_counter()-start
    times[key+"_cpu"] = times.get(key+"_cpu",0.) + time.process_time()-cpu_start
    if first is not None:
        # Stream interval includes dispatch gaps; NOT pure kernel busy time.
        times[key+"_device_stream"] = times.get(key+"_device_stream",0.) + first.elapsed_time(last)/1000
    return value


def predict(model, types, edges, families, query, qf, time_value,
            backend="upstream", context=None, token=None, times=None, details=False):
    backend_name(backend)
    device = next(model.parameters()).device.type
    if backend == "legacy":
        if context is not None: raise ValueError("legacy cannot receive tensor context")
        captured = []
        profile_handles=[]
        network_clock=[]
        if times is not None:
            def network_start(module,args):
                if device=="cuda":torch.cuda.synchronize()
                network_clock.append((time.perf_counter(),time.process_time()))
            def network_end(module,args,output):
                if device=="cuda":torch.cuda.synchronize()
                wall,cpu=network_clock.pop()
                times["network_seconds"]=times.get("network_seconds",0.)+time.perf_counter()-wall
                times["network_seconds_cpu"]=times.get("network_seconds_cpu",0.)+time.process_time()-cpu
            profile_handles=[model.backbone.register_forward_pre_hook(network_start),
                             model.backbone.register_forward_hook(network_end)]
        # Diagnostic-only hook. Ordinary training never exports probabilities.
        handle = None
        if details:
            def capture(module, args, output):
                ei = output.edge_index
                keys = ei[0]*len(types)+ei[1]
                q = torch.as_tensor(query, device=ei.device)
                idx = torch.searchsorted(keys, q[0]*len(types)+q[1])
                fam = torch.as_tensor(qf, dtype=torch.long, device=ei.device)
                captured.append(torch.stack((output.edge_attr[idx,0], output.edge_attr[idx,fam+1]),-1))
            handle = model.backbone.register_forward_hook(capture)
        try:
            p = timed(device, times, "legacy_build_transfer_network_seconds",
                      lambda: model(types,edges,families,query,qf,time_value))
        finally:
            if handle is not None: handle.remove()
            for h in profile_handles:h.remove()
        if times is not None:
            times["legacy_input_build_transfer_readout_seconds"]=max(0.,times["legacy_build_transfer_network_seconds"]-times["network_seconds"])
        logits = captured[0] if captured else torch.empty((0,2),device=device)
    else:
        owned = context is None
        token = ("single-call", float(time_value)) if owned else token
        if owned:
            context = timed(device,times,"context_seconds",lambda: FrozenTypedContext(model,types,edges,families,token))
        try:
            prepared = timed(device,times,"query_validation_seconds",lambda: context.prepare_query(query,qf,token))
            transfer = timed(device,times,"query_transfer_seconds",lambda: context.transfer_query(prepared,token))
            graph = timed(device,times,"graph_assembly_seconds",lambda: context.assemble(transfer,token))
            logits,p = timed(device,times,"network_seconds",lambda: context.run_network(graph,time_value,token))
        finally:
            if owned: context.close()
    return (logits,p) if details else p


class StepPredictor:
    def __init__(self, model, case, state, kernel, t, backend="upstream", event=None, check_time=None, profile=False):
        from hsgd.bw_preflight.common import state_edges
        self.model,self.case,self.kernel,self.t = model,case,kernel,int(t)
        self._frozen_time = self.t
        self.backend = backend_name(backend)
        self.device = next(model.parameters()).device.type
        self.times,self.blocks,self.closed = ({} if profile else None),0,False
        self.state = np.array(state,copy=True)
        self.state.flags.writeable = False
        self.edges,self.fam = timed(self.device,self.times,"state_edge_extraction_seconds",lambda: state_edges(case,self.state))
        self.token = ("reverse-time",self.t,id(self))
        self.context = None
        if backend == "upstream":
            self.context = timed(self.device,self.times,"context_seconds",lambda: FrozenTypedContext(model,case["space"].types,self.edges,self.fam,self.token))
        self.event,self.check_time = event,check_time

    def __call__(self, frozen, ids, fam, query):
        from hsgd.bw_preflight.common import check_prob
        if self.closed or self.t!=self._frozen_time or not np.array_equal(frozen,self.state):
            raise ValueError("closed or changed frozen state")
        if self.check_time: self.check_time()
        if self.event: self.event("reverse_block_start",time=self.t,block=self.blocks+1,backend=self.backend)
        with torch.no_grad():
            p = predict(self.model,self.case["space"].types,self.edges,self.fam,query,fam,
                        self.t/(len(self.kernel.alpha)-1),self.backend,self.context,self.token,self.times)
        p0 = timed(self.device,self.times,"probability_check_export_seconds",lambda: check_prob(p,fam,self.case))
        self.blocks += 1
        if self.event: self.event("reverse_block",time=self.t,block_completed=self.blocks,backend=self.backend)
        return p0

    def close(self):
        if self.context is not None: self.context.close()
        self.closed = True


def reverse_step(model,case,state,kernel,t,rounds,seed,rng,device,backend="upstream",
                 event=None,check_time=None,trace=False,interrupt_block=None):
    """Production step: no state commit here; RNG rollback on ANY partial failure.

    Caller writes/commits the returned complete state and counters atomically.
    Same original plan seed and exactly one S-wide uniform draw per time step.
    """
    from hsgd.bw_preflight.common import plan_for,rng_save,rng_restore
    from hsgd.kernel import full_sweep
    before = rng_save(rng,device)
    times, callback = ({} if trace else None),None
    if device=="cuda":torch.cuda.synchronize()
    start = time.perf_counter()
    arrays = {}
    try:
        plan = timed(device,times,"query_plan_seconds",lambda: plan_for(case,[seed,t,61],rounds))
        uniform = timed(device,times,"uniform_seconds",lambda: rng.random(case["space"].total))
        callback = StepPredictor(model,case,state,kernel,t,backend,event,check_time,profile=trace)
        coverage = np.zeros(case["space"].total,dtype=np.uint8)
        if trace:
            arrays = dict(p0=np.zeros(len(coverage)),q=np.zeros(len(coverage)),uniform=uniform.copy())
        def predict_clean(frozen,ids,fam,query):
            value = callback(frozen,ids,fam,query)
            coverage[ids] += 1
            if trace:
                arrays["p0"][ids]=value
                arrays["q"][ids]=kernel.reverse(value,frozen[ids],fam,t-1,t)
            if interrupt_block is not None and callback.blocks == interrupt_block:
                raise InterruptedError("injected engineering interruption before full-step commit")
            return value
        class TimedKernel:
            prior = kernel.prior
            def reverse(self,*args):
                return timed(device,times,"posterior_seconds",lambda: kernel.reverse(*args))
        value = timed(device,times,"full_sweep_seconds",lambda: full_sweep(plan,state,TimedKernel(),predict_clean,uniform,t-1,t))
        if times is not None: times.update(callback.times or {})
        if device=="cuda":torch.cuda.synchronize()
        expected = np.repeat(case["active"].astype(np.uint8),case["space"].counts)
        if not np.array_equal(coverage,expected): raise AssertionError("coverage incorrect")
        arrays.update(state=value.astype(np.uint8),coverage=coverage)
        return value.astype(np.uint8),dict(time=t,remaining_time=t-1,backend=backend,
                blocks=callback.blocks,covered_slots=int(coverage.sum()),exact_coverage=True,
                step_seconds=time.perf_counter()-start,timings=times or {},
                timing_scope=("synchronized detailed step; trace adds diagnostic posterior/copy overhead" if trace
                              else "low-interference end-to-end; device synchronized at step boundaries; original checks/logs retained"),
                **({"trace":arrays} if trace else {}))
    except BaseException:
        rng_restore(before,rng,device)
        raise
    finally:
        if callback is not None: callback.close()
