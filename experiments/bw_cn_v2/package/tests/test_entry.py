"""Bounded CPU native entry/restore checks; NEVER formal IMDB learning."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import pytest

ROOT=Path(__file__).resolve().parents[1]


def env():return dict(os.environ,OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")


def test_check_fixture_cannot_authorize_formal(tmp_path):
    from hsgd.cn_checks import run_check
    from hsgd.cn_entry import verify_check
    out=tmp_path/"fixture_check"
    r=run_check(out,"cpu",True)
    assert r["status"]=="passed" and r["engineering_fixture"] is True
    with pytest.raises(RuntimeError,match="fixture"):verify_check(out,"cpu")
    (out/"mechanism.json").unlink()
    with pytest.raises(RuntimeError):verify_check(out,"cpu")


def test_actual_native_stop_resume_and_independent_audit(tmp_path):
    out=tmp_path/"cn_native"
    firstlog=tmp_path/"first.log"
    command=[sys.executable,"-m","hsgd.cn_entry","start","--fixture","--arm","cn",
        "--output",str(out),"--max-updates","6"]
    with firstlog.open("w") as f:
        proc=subprocess.Popen(command,cwd=ROOT,env=env(),stdout=f,stderr=subprocess.STDOUT)
        deadline=time.monotonic()+160;requested=False
        try:
            while proc.poll() is None:
                p=out/"progress.json"
                if p.exists() and not requested:
                    row=json.loads(p.read_text())
                    if row["completed_updates"]>=1:
                        (out/"STOP_REQUESTED").write_text("CPU test safe boundary stop\n");requested=True
                if time.monotonic()>deadline:raise TimeoutError("bounded native entry test")
                time.sleep(.02)
            assert proc.returncode==0,firstlog.read_text()[-5000:]
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=20)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
    first=json.loads((out/"summary.json").read_text())
    assert requested and first["status"]=="stopped_manual" and first["formal_training"] is False
    assert 1<=first["completed_updates"]<6
    original=(out/"summary.json").read_bytes()
    wrong=subprocess.run([sys.executable,"-m","hsgd.cn_entry","resume","--fixture","--arm","baseline","--output",str(out)],
        cwd=ROOT,env=env(),capture_output=True,text=True,timeout=30)
    assert wrong.returncode!=0 and (out/"summary.json").read_bytes()==original
    command=[sys.executable,"-m","hsgd.cn_entry","resume","--fixture","--arm","cn","--output",str(out)]
    r=subprocess.run(command,cwd=ROOT,env=env(),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=120)
    assert r.returncode==0,r.stdout.decode()[-5000:]
    final=json.loads((out/"summary.json").read_text())
    assert final["status"]=="stopped_limit" and final["completed_updates"]==6 and not final["formal_training"]
    assert list(out.glob("resume_loaded_*.json"))
    r=subprocess.run([sys.executable,"-m","hsgd.learning_verify",str(out),"--arm","cn"],
        cwd=ROOT,env=env(),capture_output=True,text=True,timeout=60)
    assert r.returncode==0,r.stdout+r.stderr
    assert json.loads(r.stdout)["verified"]
    # Versions/config/arm bindings reject a baseline resume of a CN checkpoint.
    r=subprocess.run([sys.executable,"-m","hsgd.cn_entry","resume","--fixture","--arm","baseline","--output",str(out)],
        cwd=ROOT,env=env(),capture_output=True,text=True,timeout=30)
    assert r.returncode!=0


def test_native_baseline_independent_entry(tmp_path):
    out=tmp_path/"baseline_native"
    r=subprocess.run([sys.executable,"-m","hsgd.cn_entry","start","--fixture","--arm","baseline",
        "--output",str(out),"--max-updates","2"],cwd=ROOT,env=env(),capture_output=True,text=True,timeout=160)
    assert r.returncode==0,r.stdout[-5000:]+r.stderr[-5000:]
    s=json.loads((out/"summary.json").read_text());f=json.loads((out/"freeze.json").read_text())
    assert s["completed_updates"]==2 and not s["formal_training"] and s["status"]=="stopped_limit"
    assert f["actual_dimensions"]["model"]["common_neighbor_parameters"]==0
    r=subprocess.run([sys.executable,"-m","hsgd.learning_verify",str(out),"--arm","baseline"],
        cwd=ROOT,env=env(),capture_output=True,text=True,timeout=60)
    assert r.returncode==0,r.stdout+r.stderr
