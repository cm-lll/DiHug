"""One nonrenewable 60-minute CPU window, including checks and final audit."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import hashlib

ROOT=Path(__file__).resolve().parent


def save(p,x):
    tmp=p.with_suffix(".tmp");tmp.write_text(json.dumps(x,indent=2)+"\n");tmp.replace(p)


def activate(out,seconds=3600):
    # Exclusive sibling ledger prevents using the same output with a new clock.
    ledger=out.with_name(out.name+".budget.json")
    if out.exists():raise FileExistsError(out)
    epoch=time.time()
    with ledger.open("x") as f:
        json.dump(dict(activation_epoch=epoch,hard_cutoff_epoch=epoch+seconds,budget_seconds=seconds,
                       no_reset=True,output=str(out.resolve()),scope="checks,training,sampling,evaluation,saving,audit"),f,indent=2)
    out.mkdir();return ledger,epoch


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--continue-preflight-fix",action="store_true",
                    help="only failed pre-science checks, same immutable deadline; never resume learning")
    a=ap.parse_args();a.output=a.output.resolve();a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.continue_preflight_fix:
        ledger=a.output.with_name(a.output.name+".budget.json")
        old=json.loads((a.output/"budget_receipt.json").read_text());budget=json.loads(ledger.read_text())
        assert old["status"]=="failed" and all(j["job"] in ("freeze","tests") for j in old["jobs"])
        assert not (a.output/"updates.jsonl").exists() and not (a.output/"cost_gate.json").exists()
        assert old["ledger_sha256"]==hashlib.sha256(ledger.read_bytes()).hexdigest()
        assert budget["hard_cutoff_epoch"]>time.time(),"original budget expired"
        archive=a.output/"preflight_attempt_01";archive.mkdir(exist_ok=False)
        for p in list(a.output.iterdir()):
            if p.is_file() and p.name!="STOP":p.rename(archive/p.name)
        epoch=budget["activation_epoch"]
        start=time.monotonic()-(time.time()-epoch);deadline=start+3600
        save(a.output/"preflight_fix.json",dict(reason="test import resolved baseline/run.py instead of this launcher",
             prior_failed_tests_preserved=True,original_activation_epoch=epoch,
             original_hard_cutoff_epoch=budget["hard_cutoff_epoch"],budget_reset=False))
    else:
        ledger,epoch=activate(a.output);start=time.monotonic();deadline=start+3600
    env=dict(os.environ,CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",
             PYTEST_ADDOPTS="",PYTHONPATH=str(ROOT.parent/"role_condition_diagnosis_v1/baseline"))
    jobs=[];child=None;status="failed";reason=None
    def stop(sig,frame):raise InterruptedError(f"signal {sig}")
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,stop)
    def job(name,command):
        nonlocal child
        with (a.output/(name+".log")).open("w") as log:
            begin=time.monotonic();child=subprocess.Popen(command,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while child.poll() is None:
                if (a.output/"STOP").exists():raise InterruptedError("STOP")
                if time.monotonic()>deadline-5:raise TimeoutError("single cumulative budget exhausted")
                time.sleep(.5)
            row=dict(job=name,exit_code=child.returncode,seconds=time.monotonic()-begin)
            jobs.append(row);save(a.output/"launcher_progress.json",row);print(json.dumps(row),flush=True)
            if child.returncode:raise RuntimeError(f"{name} exited {child.returncode}")
    try:
        # Binding is frozen before all executable tests and scientific work.
        job("freeze",[sys.executable,str(ROOT/"worker.py"),"freeze",str(a.output),str(deadline)])
        previous=ROOT.parent/"role_condition_diagnosis_v1"
        job("tests",[sys.executable,"-m","pytest","-q",str(previous/"baseline/tests"),
            str(previous/"test_late_fusion.py"),str(ROOT/"test_pairing.py"),
            "--junitxml="+str(a.output/"tests.xml")])
        import xml.etree.ElementTree as ET
        suites=ET.parse(a.output/"tests.xml").getroot().findall("testsuite")
        assert sum(int(s.attrib["tests"]) for s in suites)==81
        assert all(int(s.attrib[k])==0 for s in suites for k in ["errors","failures","skipped"])
        job("panel",[sys.executable,str(ROOT/"worker.py"),"panel",str(a.output),str(deadline)])
        job("audit",[sys.executable,str(ROOT/"audit.py"),str(a.output),str(deadline)])
        status="completed"
    except BaseException as exc:
        reason=repr(exc);status="stopped" if isinstance(exc,(InterruptedError,TimeoutError)) else "failed"
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=3)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=2)
        save(a.output/"budget_receipt.json",dict(status=status,reason=reason,jobs=jobs,
             activation_epoch=epoch,end_epoch=time.time(),wall_seconds=time.monotonic()-start,
             budget_seconds=3600,no_reset=True,ledger_sha256=hashlib.sha256(ledger.read_bytes()).hexdigest(),
             formal_large_graph_runs=0,notebook_BW_access=0))
    return 0 if status=="completed" else 1
if __name__=="__main__":sys.exit(main())
