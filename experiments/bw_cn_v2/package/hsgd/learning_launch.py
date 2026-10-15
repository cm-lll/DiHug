"""CN v2: v9 bounded strict preflight -> ordinary from-scratch IMDB process.

No eight-hour window or inherited deadline. Default maximum10000 updates.
This launcher never starts a complete generation chain.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from hsgd.learning_budget import atomic_json
from hsgd.learning_control import open_control,elapsed


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--resume",action="store_true")
    p.add_argument("--arm",choices=("baseline","cn"),required=True)
    p.add_argument("--check-dir",type=Path,help="passed independent BW engineering/cost receipt")
    stop=p.add_mutually_exclusive_group()
    stop.add_argument("--manual-stop",action="store_true",help=argparse.SUPPRESS)
    stop.add_argument("--stop-after-updates",type=int,help="optional USER-selected total successful update limit")
    p.add_argument("--related-failure-run",type=Path,help="original failed v7 output, provenance only; old expiry ignored")
    p.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    p.add_argument("--data",type=Path,default=Path(__file__).resolve().parents[1]/"data/IMDB_movie_reference.npz")
    p.add_argument("--fixture",action="store_true",help="CPU engineering only, never real-data learning")
    p.add_argument("--preflight-timeout-seconds",type=int,default=600)
    args=p.parse_args();out=args.output.resolve()
    from hsgd.bw_preflight.common import set_arm
    set_arm(args.arm)
    if args.manual_stop:p.error("this experiment requires a maximum update count; default10000")
    if not args.resume and args.stop_after_updates is None:args.stop_after_updates=10000
    if not 30<=args.preflight_timeout_seconds<=600:p.error("preflight timeout must be30..600s")
    if args.fixture and args.device!="cpu":p.error("fixture CPU only")
    if not args.fixture and (args.device!="cuda" or not str(out).startswith("/root/private_data/")):
        p.error("formal BW cuda artifacts must be under persistent /root/private_data")
    if args.resume:
        if json.loads((out/"freeze.json").read_text())["binding"]["config"]["experiment_arm"]!=args.arm:
            p.error("resume arm differs; rejecting before any run mutation")
    control=open_control(out,args.resume,args.manual_stop,args.stop_after_updates,args.related_failure_run,args.fixture)
    if not args.fixture and not args.resume:
        from hsgd.cn_entry import prepare_runtime,verify_check
        prepare_runtime(args.device)
        verified=verify_check(args.check_dir,args.device)
        control["cn_engineering_check"]=verified
        atomic_json(out/"run_control.json",control)
    elif not args.fixture:
        from hsgd.cn_entry import prepare_runtime,verify_check
        prepare_runtime(args.device)
        if verify_check(Path(control["cn_engineering_check"]["directory"]),args.device)!=control["cn_engineering_check"]:
            raise RuntimeError("native engineering receipt changed")
    lock=(out/".launcher.lock").open("a+");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if args.resume:
        if not (out/"latest.json").is_file():raise RuntimeError("no native formal checkpoint; preflight failures require review")
        if (out/"summary.json").exists():
            old=json.loads((out/"summary.json").read_text())
            if old["status"] in ("failed","stopped_limit"):raise RuntimeError("failed/finished run requires review; not automatic continuation")
        request=out/"STOP_REQUESTED"
        if request.exists():request.rename(out/("stop_request_archived_%d.json"%time.time_ns()))
    if not args.fixture and shutil.disk_usage(out).free<3*2**30:raise RuntimeError("need at least3GiB free for checkpoints/cache/temp files")
    atomic_json(out/("invocation_%d.json"%time.time_ns()),dict(command=sys.argv,python=sys.version,executable=sys.executable,
        resume=args.resume,control=control,preflight_timeout_seconds=args.preflight_timeout_seconds))
    print(json.dumps(dict(output=str(out),stop_mode=control["stop_mode"],stop_after_updates=control["stop_after_updates"],
        eight_hour_limit=False,hard_cutoff_utc=None,automatic_generation=False,
        preflight_strict=True,training_strict=False)),flush=True)
    started=time.time();launches=[];interrupted=[False];active=[None]
    atomic_json(out/"launcher_pid.json",dict(pid=os.getpid(),created_epoch=started,output=str(out)))
    def interrupt(signum,frame):
        interrupted[0]=True
        if active[0] is not None:
            try:os.killpg(active[0].pid,signal.SIGINT)
            except ProcessLookupError:pass
    signal.signal(signal.SIGINT,interrupt);signal.signal(signal.SIGTERM,interrupt)
    def launch(job,command,cap=None):
        log=(out/(job+"_%d.log"%time.time_ns())).open("w")
        print("START",job,"timeout_seconds",cap,"mode","ordinary" if job=="train" else "strict",flush=True)
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);active[0]=child
        tick=time.time()
        atomic_json(out/"active_worker.json",dict(job=job,pid=child.pid,expected_strict=job!="train"))
        try:
            if cap is None:code=child.wait()  # No implicit formal-training timer.
            else:
                try:code=child.wait(timeout=cap)
                except subprocess.TimeoutExpired:
                    try:os.killpg(child.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    child.wait();code=75
        finally:active[0]=None;log.close()
        row=dict(job=job,exit_code=code,pid=child.pid,run_elapsed_seconds=elapsed(control),
                 child_wall_seconds=time.time()-tick,log=log.name,strict=job!="train")
        launches.append(row);atomic_json(out/"launcher_progress.json",row)
        print("END",job,"exit",code,flush=True);return code
    flags=["--device",args.device,"--data",str(args.data.resolve()),"--arm",args.arm]
    if args.fixture:flags.append("--fixture")
    code=0
    if not args.resume:
        deadline=time.monotonic()+args.preflight_timeout_seconds
        for job in ("gate_initialize","training_pair","training_continuous","training_resume"):
            if (out/"STOP_REQUESTED").exists() or interrupted[0]:code=130;break
            remain=deadline-time.monotonic()
            if remain<=1:code=75;break
            if job=="gate_initialize":
                command=[sys.executable,"-m","hsgd.learning","--phase",job,"--output",str(out/"preflight"),*flags]
            else:
                command=[sys.executable,"-m","hsgd.runtime_gate","--output",str(out/"preflight"),"--job",job,
                    "--worker-cap",str(min(540,max(1,remain-5))),*flags]
            code=launch(job,command,min(570,remain))
            if code:break
    if not code and not interrupted[0] and not (out/"STOP_REQUESTED").exists():
        command=[sys.executable,"-m","hsgd.learning","--phase","train","--output",str(out),*flags]
        if args.resume:command.append("--resume")
        code=launch("train",command)  # Training worker enforces ONLY the explicit stop policy.
    elif not code:code=130
    receipt=dict(status="completed" if code==0 else "incomplete" if code in (75,130) else "failed",
        child_exit_code=code,jobs=launches,control=control,run_elapsed_seconds=elapsed(control),
        formal_worker_launched=any(x["job"]=="train" for x in launches),no_automatic_generation=True,
        formal_timeout_seconds=None,preflight_timeout_seconds=args.preflight_timeout_seconds,
        accelerator_process_wall_upper_hours=sum(x["child_wall_seconds"] for x in launches)/3600 if args.device=="cuda" else 0)
    atomic_json(out/("launcher_receipt_%d.json"%time.time_ns()),receipt)
    summary=out/"summary.json"
    if code and (not summary.exists() or summary.stat().st_mtime<started):
        previous=json.loads(summary.read_text()) if summary.exists() else None
        if previous:summary.rename(out/("summary_previous_%d.json"%time.time_ns()))
        atomic_json(summary,dict(stage="cn-v2-IMDB-B-learning",status="incomplete" if code in (75,130) else "failed",
            reason="preflight/worker incomplete; no gate waiver",completed_updates=previous.get("completed_updates") if previous else None,
            run_control=control,final_checkpoint=None,launcher=receipt))
    print(json.dumps(receipt,indent=2),flush=True)
    return code


if __name__=="__main__":sys.exit(main())
