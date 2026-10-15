"""Explicit stop policy and provenance; no eight-hour budget or deadline."""
import hashlib
import json
from pathlib import Path
import time
from hsgd.learning_budget import atomic_json,utc,signature


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()


def open_control(out,resume=False,manual=False,stop_updates=None,related=None,fixture=False):
    out=Path(out)
    if resume:
        if manual or stop_updates is not None or related:raise RuntimeError("resume uses frozen stop policy/provenance")
        c=json.loads((out/"run_control.json").read_text())
        if c["version"]!="bw_cn_v1" or c["engineering_fixture"]!=fixture:raise RuntimeError("run control version/mode mismatch")
        validate(c)
        return c
    if manual==bool(stop_updates is not None):raise RuntimeError("select --manual-stop OR --stop-after-updates N explicitly; no default duration")
    if stop_updates is not None and stop_updates<1:raise RuntimeError("stop updates must be positive")
    if out.exists():raise RuntimeError("new independent output required")
    files={}
    if related:
        related=Path(related).resolve()
        summary=json.loads((related/"summary.json").read_text())
        if summary.get("status")!="failed":raise RuntimeError("related failure record is not failed; never rewrite its status")
        for name in ("summary.json","preflight/run.json","preflight/initial.pt","preflight/training_pair_input.pt"):
            files[str(related/name)]=sha(related/name)
    # A new, from-scratch experiment has no dependency on the failed old v7 run.
    now=time.time()
    c=dict(version="bw_cn_v1",created_epoch=now,created_utc=utc(now),engineering_fixture=fixture,
        eight_hour_limit=False,hard_cutoff_utc=None,stop_mode="manual" if manual else "updates",
        stop_after_updates=stop_updates,related_failure_files=files,
        preflight_mode=dict(deterministic_algorithms=True,deterministic_warn_only=False),
        training_mode=dict(deterministic_algorithms=False,deterministic_warn_only=False),
        validation_mode=dict(deterministic_algorithms=False,deterministic_warn_only=False),
        resumed_trajectory_identity_required=False,automatic_generation=False,
        stop_request_filename="STOP_REQUESTED",default_without_stop_choice="refuse before model launch")
    out.mkdir(parents=True);atomic_json(out/"run_control.json",c);return c


def validate(c):
    if c["eight_hour_limit"] is not False or c["hard_cutoff_utc"] is not None:raise RuntimeError("unexpected time budget/deadline")
    if c["stop_mode"] not in ("manual","updates") or ((c["stop_mode"]=="manual")!=(c["stop_after_updates"] is None)):
        raise RuntimeError("invalid explicit stop policy")
    if c["stop_mode"]=="updates" and c["stop_after_updates"]<1:raise RuntimeError("invalid update stop")
    for key,strict in (("preflight_mode",True),("training_mode",False),("validation_mode",False)):
        if c[key]!=dict(deterministic_algorithms=strict,deterministic_warn_only=False):raise RuntimeError("execution mode control changed")
    for path,digest in c["related_failure_files"].items():
        if sha(path)!=digest:raise RuntimeError("related failure source changed: "+path)


def elapsed(c):return max(0.,time.time()-c["created_epoch"])


def stop_reason(out,c,completed):
    if (Path(out)/"STOP_REQUESTED").exists():return "manual_stop_request"
    if c["stop_mode"]=="updates" and completed>=c["stop_after_updates"]:return "configured_update_limit"
    return None
