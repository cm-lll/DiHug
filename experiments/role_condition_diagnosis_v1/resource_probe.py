"""Explicit full-IMDB resource gate; temporary engineering weights only.

Parent imports no Torch. 570s includes startup; cleanup has a separate 5s cap.
Never loads a checkpoint, saves trained weights, or runs more than one reverse
step per model. No resource/config fallback on failure. GPU UUID required.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback


def save(path,value):
    tmp=path.with_suffix(".tmp");tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
    tmp.replace(path)


def worker(args):
    import hashlib
    import resource
    import gc
    import numpy as np
    import torch
    sys.path.insert(0,str(Path(__file__).parent/"baseline"))
    from hsgd.bw_preflight import common as c
    from hsgd.kernel import MarginalKernel
    from hsgd.runtime import reverse_step
    from role_condition.model import ConditionalDenoiser, extract_target
    from late_fusion import LateConditionalDenoiser
    out=args.output;start=time.perf_counter();cpu=time.process_time();rows=[]
    phase="initialize";status="failed"
    try:
        torch.set_num_threads(1);torch.set_float32_matmul_precision("highest")
        torch.use_deterministic_algorithms(False)
        assert torch.cuda.is_available() and torch.cuda.device_count()==1
        torch.cuda.set_device(0)
        config=json.loads((Path(__file__).parent/"baseline/role_condition/config.json").read_text())
        data_config=c.config()
        case=c.imdb_case(args.data,data_config);target=extract_target(case)
        T,R=1000,20
        inputs=[c.train_input(case,T,R,2026102041,i,np.random.default_rng([2026102041,i]),
                              force_t=t) for i,t in enumerate([1,500,1000,500,500])]
        plan=c.plan_for(case,[2026102041,500,61],R)
        binding=dict(data_sha256=c.sha(args.data),nodes=len(case["space"].types),edges=len(case["clean_ids"]),
            family_edges=case["counts"].tolist(),schema=case["space"].pairs,
            inventory=c.query_inventory(case,plan),environment=c.environment("cuda"),
            model=config["model"],optimizer=config["optimizer"],T=T,rounds=R,
            seed=2026102041,weights="fresh temporary engineering initialization; no saved/BW model imported",
            cpu_threads=1,device_uuid=args.gpu_uuid,target=target.as_dict(),
            numerical_contract="FP32 network; float64 kernel; no autocast; no TF32",
            source_hashes={str(p.relative_to(Path(__file__).parent)):c.sha(p) for p in Path(__file__).parent.rglob("*.py")})
        save(out/"binding.json",binding)
        # Construct each model on CPU; explicitly copy all compatible common
        # parameters from identical initial C. No old trained C native restore.
        c.seed_all(2026102041)
        original=ConditionalDenoiser(case["n_types"],case["n_families"],family_pairs=case["space"].pairs,
            prior=case["prior"],mode="C",**config["model"])
        common={k:v.detach().clone() for k,v in original.state_dict().items()}
        del original
        for label,cls in [("original_C",ConditionalDenoiser),("late_C",LateConditionalDenoiser)]:
            phase=label+"_initialize";print(phase,flush=True)
            model=cls(case["n_types"],case["n_families"],family_pairs=case["space"].pairs,
                      prior=case["prior"],mode="C",**config["model"])
            state=model.state_dict()
            for k,v in common.items():
                if k in state:state[k]=v.clone()
            model.load_state_dict(state,strict=True);model.set_target(target);model.to("cuda")
            assert all(p.dtype==torch.float32 for p in model.parameters())
            optcfg={k:v for k,v in config["optimizer"].items() if k!="type"}
            opt=torch.optim.AdamW(model.parameters(),**optcfg)
            torch.cuda.reset_peak_memory_stats()
            for i,inp in enumerate(inputs):
                phase=label+f"_update{i}";torch.cuda.synchronize();tick=time.perf_counter()
                result=c.update(model,opt,case,inp,T,"cuda")
                torch.cuda.synchronize();elapsed=time.perf_counter()-tick
                free,total=torch.cuda.mem_get_info()
                item=dict(model=label,operation="engineering_update",index=i,t=inp["t"],
                          timing_class="warmup" if i<2 else "steady",wall_seconds=elapsed,
                          device_total_used_mib=(total-free)/2**20,**result)
                rows.append(item);save(out/"measurements.json",rows)
                print(json.dumps(item),flush=True)
            # One full, frozen-state, 20-block step. It is a forward-noised
            # reference fixture, NOT a state from a prior generation chain.
            phase=label+"_reverse500";model.eval();inp=inputs[3]
            f,loc=case["space"].encode(inp["edges"]);state=np.zeros(case["space"].total,dtype=np.uint8)
            state[case["space"].offsets[f]+loc]=1
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();tick=time.perf_counter()
            with torch.no_grad():
                value,info=reverse_step(model,case,state,MarginalKernel(case["prior"],c.schedule(T)),
                    500,R,2026102041,np.random.default_rng(2026102042),"cuda",trace=False)
            torch.cuda.synchronize();elapsed=time.perf_counter()-tick
            free,total=torch.cuda.mem_get_info()
            rows.append(dict(model=label,operation="complete_20_block_reverse_step",wall_seconds=elapsed,
                peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                device_total_used_mib=(total-free)/2**20,**info))
            save(out/"measurements.json",rows)
            print(json.dumps(rows[-1]),flush=True)
            del model,opt,value,state;gc.collect();torch.cuda.empty_cache()
        status="completed"
    except BaseException as exc:
        save(out/"failure.json",dict(phase=phase,type=type(exc).__name__,message=str(exc),
                                    traceback=traceback.format_exc()))
        raise
    finally:
        save(out/"worker_receipt.json",dict(status=status,phase=phase,wall_seconds=time.perf_counter()-start,
            cpu_seconds=time.process_time()-cpu,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            updates=sum(x["operation"]=="engineering_update" for x in rows),full_chains=0,
            note="all temporary engineering weights discarded; no scientific graph output or compatibility claim"))


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--data",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True);ap.add_argument("--gpu-uuid",required=True)
    ap.add_argument("--worker",action="store_true");a=ap.parse_args()
    if a.worker:return worker(a)
    start=time.monotonic()
    # Respect scheduler isolation: only select from already visible IDs/UUIDs.
    visible=os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and a.gpu_uuid not in visible.split(","):
        raise RuntimeError("Existing CUDA visibility does not explicitly include requested UUID; no isolation override")
    query=subprocess.check_output(["nvidia-smi","--id="+a.gpu_uuid,
        "--query-gpu=uuid,name,memory.total,memory.used,utilization.gpu","--format=csv,noheader,nounits"],text=True,timeout=15)
    cols=[x.strip() for x in query.strip().split(",")]
    if int(cols[3])>32 or int(cols[4])>1:raise RuntimeError("selected device not idle")
    apps=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid","--format=csv,noheader"],text=True,timeout=15)
    if a.gpu_uuid in apps:raise RuntimeError("selected device has an existing compute process")
    a.output.mkdir(parents=True,exist_ok=False)
    save(a.output/"activation.json",dict(device_query=query,compute_apps=apps,cap_seconds=570,
         cleanup_cap_seconds=5,existing_visibility=visible,scope="local explicit one-GPU engineering resource gate"))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpu_uuid,OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
    reason="failed";child=None;rc=1
    def stop(sig,frame):raise InterruptedError(f"signal {sig}")
    for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(sig,stop)
    try:
        with (a.output/"worker.log").open("w") as log:
            child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),"--worker","--data",str(a.data),
                  "--output",str(a.output),"--gpu-uuid",a.gpu_uuid],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while child.poll() is None:
                if time.monotonic()-start>=570:reason="timeout";raise TimeoutError(reason)
                if (a.output/"STOP").exists():reason="STOP";raise InterruptedError(reason)
                time.sleep(.2)
            rc=child.returncode;reason="completed" if rc==0 else "failed"
    except InterruptedError as exc:
        if reason!="STOP":reason=str(exc)
        rc=130
    except TimeoutError:rc=124
    finally:
        cleanup=time.monotonic()
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=2)
        save(a.output/"controller_receipt.json",dict(status=reason,exit_code=rc,
             total_wall_seconds=time.monotonic()-start,cleanup_seconds=time.monotonic()-cleanup,
             cap_seconds=570,cleanup_extra_cap_seconds=5))
    return rc
if __name__=="__main__":sys.exit(main())
