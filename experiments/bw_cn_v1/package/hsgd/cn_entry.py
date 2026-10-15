"""Thin notebook entry over v9 learning, validation, transactional sampling."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode=True


def prepare_runtime(device):
    import torch
    from hsgd.learning_mode import configure
    torch.set_num_threads(1);torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    configure(False)
    from hsgd.bw_preflight.common import environment
    return environment(device)


def verify_check(directory,device):
    from hsgd.bw_preflight import common as c
    if directory is None:raise RuntimeError("run check first; --check-dir required")
    directory=Path(directory).resolve();r=json.loads((directory/"check.json").read_text())
    if r["status"]!="passed" or r["engineering_fixture"] is not False or not r["checks"]:
        raise RuntimeError("BW engineering/cost gate incomplete or fixture only")
    if r["code"]!=c.code_hashes() or r["data_sha256"]!=c.config()["imdb_sha256"]:
        raise RuntimeError("engineering receipt code/data changed")
    if c.runtime_binding(r["environment"])!=c.runtime_binding(c.environment(device)):
        raise RuntimeError("engineering runtime differs")
    for n in ("mechanism.json","baseline_cost.json","cn_cost.json"):
        if n not in r["files"] or c.sha(directory/n)!=r["files"][n]:raise RuntimeError("engineering evidence missing/changed")
    mechanism=json.loads((directory/"mechanism.json").read_text())
    if mechanism["status"]!="passed" or len(mechanism["checks"])<4 or mechanism["small_reverse_steps"]!=4:
        raise RuntimeError("mechanism/short-chain evidence incomplete")
    for arm in ("baseline","cn"):
        a=r["arms"][arm]
        if a["measured_updates"]!=2 or len(a["updates"])!=3 or a["full_20_block_step"]["blocks"]!=20 or not a["full_20_block_step"]["exact_coverage"]:
            raise RuntimeError("incomplete full-size cost evidence")
        if a["reference"]["nodes"]!=c.config()["imdb_nodes"] or a["reference"]["edges"]!=c.config()["imdb_edges"]:
            raise RuntimeError("cost measurement is not the full reference")
        expected=10116 if arm=="cn" else 0
        if a["dimensions"]["common_neighbor_parameters"]!=expected:raise RuntimeError("wrong arm parameter count")
    return dict(directory=str(directory),sha256=c.sha(directory/"check.json"),files=r["files"])


def load_final(run,arm,device):
    import numpy as np
    import torch
    from hsgd.bw_preflight import common as c
    from hsgd.learning_budget import model_fingerprint,signature
    from hsgd.learning_mode import require
    from hsgd.learning import PROTOCOL
    from hsgd.cn_sampling import required,digest
    c.set_arm(arm);c.verify_manifest();cfg=c.config();env=prepare_runtime(device)
    if device!="cuda" or env["device_name"]!="BW" or not env["hip"]:raise RuntimeError("formal sampler requires BW")
    run=Path(run).resolve()
    names=("summary.json","freeze.json","latest.json","run_control.json")
    s,f,l,control=[json.loads(required(run,n).read_text()) for n in names]
    if s.get("stage")!="cn-v1-IMDB-B-learning" or s.get("formal_training") is not True or s.get("status") not in ("stopped_manual","stopped_limit"):
        raise RuntimeError("not a finalized formal CN-experiment run")
    if control.get("version")!="bw_cn_v1" or control.get("engineering_fixture") is not False:raise RuntimeError("old/fixture checkpoint rejected")
    path=required(run,s.get("final_checkpoint"))
    if s["final_checkpoint"]!=l["path"] or digest(path)!=l["sha256"]:raise RuntimeError("final/latest/hash mismatch")
    ck=c.trusted_load(path);b=f["binding"];n=s["completed_updates"]
    if not (type(n) is int and n>0 and n==l["completed_updates"]==ck["completed_updates"]):raise RuntimeError("effective update counts differ")
    if (b["code"]!=c.code_hashes() or b["config"]!=cfg or b["execution_protocol"]!=PROTOCOL or
        b["graph_backend"]!="upstream" or b["run_control"]!=control or ck["binding"]!=b or s["run_control"]!=control):
        raise RuntimeError("source/config/arm/protocol binding mismatch")
    require(c.runtime_binding(env),False)
    if c.runtime_binding(env)!=b["backend"]:raise RuntimeError("dependency/runtime binding mismatch")
    source=c.PACKAGE/"data/IMDB_movie_reference.npz";provenance=c.PACKAGE/"data/IMDB_preprocessing_provenance.json"
    if c.sha(source)!=b["data"] or c.sha(provenance)!=b["data_provenance_sha256"]:raise RuntimeError("data binding mismatch")
    if model_fingerprint(ck["model"])!=ck["model_sha256"] or not all(torch.isfinite(x).all() for x in ck["model"].values()):
        raise RuntimeError("weight fingerprint/nonfinite")
    from hsgd.cn_optimizer import verify_steps
    verify_steps(ck)
    vals=ck["validations"]
    if vals!=s["validation_records"] or not vals or vals[-1]["stage"]!="final" or vals[-1]["status"]!="completed" or vals[-1]["completed_updates"]!=n or vals[-1]["model_sha256"]!=ck["model_sha256"]:
        raise RuntimeError("final validation/weight mismatch")
    case=c.imdb_case(source,cfg);model=c.make_model(case,cfg,device)
    model.load_state_dict(ck["model"],strict=True);model.eval()
    if model_fingerprint(model.state_dict())!=ck["model_sha256"]:raise RuntimeError("strict load differs")
    if not np.array_equal(model.prior.detach().cpu().numpy(),case["prior"]):raise RuntimeError("final prior differs from bound reference")
    manifest=json.loads((c.PACKAGE/"manifest_cn_v1.json").read_text())
    receipt=dict(formal_training=True,experiment_arm=arm,model_sha256=ck["model_sha256"],completed_updates=n,
        final_checkpoint=str(path),checkpoint_sha256=l["sha256"],config=cfg,environment=env,
        source_package=dict(path=str(c.PACKAGE),files={k:v["sha256"] for k,v in manifest["files"].items()}),
        run=str(run),run_evidence={name:digest(run/name) for name in names},binding_signature=signature(b))
    receipt["run_evidence"][s["final_checkpoint"]]=digest(path)
    return model,case,cfg,receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=("check","_check","start","resume","progress","stop","sample"))
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--arm",choices=("baseline","cn"),default="cn")
    p.add_argument("--check-dir",type=Path)
    p.add_argument("--max-updates",type=int,default=10000)
    p.add_argument("--run",type=Path)
    p.add_argument("--seed",type=int,default=2026101601)
    p.add_argument("--sample-resume",action="store_true")
    p.add_argument("--check-only",action="store_true")
    p.add_argument("--fixture",action="store_true",help=argparse.SUPPRESS)
    args=p.parse_args();out=args.output.resolve()
    if args.command in ("progress","stop"):
        if not out.is_dir():p.error("run directory missing")
        if (out/"run_control.json").exists():
            if json.loads((out/"run_control.json").read_text())["version"]!="bw_cn_v1":p.error("not a CN experiment run")
        elif (out/"sampling_binding.json").exists():
            if json.loads((out/"sampling_binding.json").read_text())["input"]["config"]["version"]!="bw_cn_v1":p.error("not a CN experiment sample")
        else:p.error("no native experiment binding")
        if args.command=="stop":
            with (out/"STOP_REQUESTED").open("a") as f:f.write("user safe stop\n")
            print("Requested safe stop; wait for terminal status and final checkpoint.")
        for name in ("active_worker.json","launcher_progress.json","progress.json","latest.json","chain_latest.json","summary.json"):
            if (out/name).exists():
                data=json.loads((out/name).read_text())
                if name=="summary.json":data={k:data.get(k) for k in ("status","reason","completed_updates","completed_reverse_steps","remaining_time","final_checkpoint","final_graph","validation_trend")}
                print(name,json.dumps(data,ensure_ascii=False))
        vals=list((out/"validation").glob("*/validation.json"))
        if vals:
            r=json.loads(max(vals,key=lambda p:p.stat().st_mtime).read_text())
            print("validation",json.dumps(dict(stage=r["stage"],updates=r["completed_updates"],
                pooled_ce=r["pooled_ce"],positive_ce=r["positive_ce"],negative_ce=r["negative_ce"],
                times=[{k:x[k] for k in ("time","pooled_ce","positive_ce","negative_ce","expected_edges")} for x in r["times"]])))
        return 0
    from hsgd.bw_preflight import common as c
    c.set_arm(args.arm)
    device="cpu" if args.fixture else "cuda"
    if args.command=="check":
        command=[sys.executable,"-m","hsgd.cn_entry","_check","--output",str(out)]
        if args.fixture:command.append("--fixture")
        try:return subprocess.run(command,timeout=570).returncode
        except subprocess.TimeoutExpired:
            if out.exists():c.json_save(out/"timeout.json",dict(status="incomplete",seconds=570))
            return 75
    if args.command=="_check":
        from hsgd.cn_checks import run_check
        r=run_check(out,device,args.fixture);print(json.dumps(r,indent=2));return 0
    if args.command in ("start","resume"):
        if args.max_updates<1:p.error("positive maximum updates required")
        if args.command=="resume":
            frozen=json.loads((out/"freeze.json").read_text())
            if frozen["binding"]["config"]["experiment_arm"]!=args.arm:p.error("resume arm differs; original run unchanged")
            if "--max-updates" in sys.argv:p.error("resume uses the frozen maximum; cannot change it")
        command=[sys.executable,"-m","hsgd.learning_launch","--output",str(out),"--arm",args.arm]
        if args.command=="resume":command.append("--resume")
        else:command.extend(["--stop-after-updates",str(args.max_updates)])
        if args.check_dir:command.extend(["--check-dir",str(args.check_dir.resolve())])
        if args.fixture:command.extend(["--fixture","--device","cpu"])
        os.execv(sys.executable,command)
    if args.command=="sample":
        if not args.run:p.error("--run finalized new experiment run required")
        if args.fixture:p.error("formal sample CLI cannot use fixtures")
        for source in (c.PACKAGE,args.run.resolve()):
            if source==out or source in out.parents or out in source.parents:p.error("output must be isolated")
        from hsgd.cn_sampling import chain,unchanged,read,write
        if args.sample_resume:
            if not out.is_dir():p.error("no native sample directory")
            if args.check_only:p.error("check-only requires a new directory")
        else:out.mkdir(parents=True,exist_ok=False)
        lock=(out/".sample.lock").open("a+");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        model,case,cfg,receipt=load_final(args.run,args.arm,device)
        unchanged(receipt)
        write(out/("input_check_%d.json"%time.time_ns()),receipt)
        if args.check_only:
            write(out/"summary.json",dict(status="checked_only",forward_calls=0));return 0
        if args.sample_resume and (out/"summary.json").exists():
            status=read(out/"summary.json")["status"]
            if status not in ("stopped","failed"):raise RuntimeError("sample is not resumable")
            (out/"summary.json").rename(out/("summary_before_resume_%d.json"%time.time_ns()))
        code=chain(model,case,cfg,receipt,out,args.seed,device,args.sample_resume)
        try:
            unchanged(receipt)
            summary=read(out/"summary.json")
            if summary["status"]=="completed":
                import numpy as np
                from hsgd.cn_metrics import evaluate,imdb_roles
                roles=imdb_roles(read(c.PACKAGE/"data/IMDB_preprocessing_provenance.json"))
                with np.load(out/"graph.npz") as z:
                    generated=evaluate(z["types"],z["edges"].T,z["family_ids"],case["space"].pairs,roles)
                reference=evaluate(case["space"].types,case["clean"],case["space"].encode(case["clean"])[0],case["space"].pairs,roles)
                summary.update(structure_diagnostics=dict(generated=generated,reference=reference),
                    source_inputs_unchanged=True,evaluator_complete=True)
                write(out/"summary.json",summary)
        except BaseException as exc:
            summary=read(out/"summary.json")
            summary.update(status="incomplete_evaluation_or_integrity",evaluator_complete=False,error=str(exc))
            write(out/"summary.json",summary)
            raise
        return code


if __name__=="__main__":
    try:raise SystemExit(main())
    except Exception:
        import traceback
        traceback.print_exc();raise SystemExit(1)
