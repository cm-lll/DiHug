"""Fail-closed independent evidence reader for v6 runtime gate, not status echo."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
from hsgd.bw_preflight.common import (config,code_hashes,sha,trusted_load,runtime_binding,
    imdb_case,toy_case,graph_stats,tree_error,PACKAGE)
from hsgd.perf_checks import compare_float,compare_rng,TOLERANCES


def need(out,names):
    for name in names:
        if not (out/name).is_file() or (out/name).stat().st_size==0:
            raise AssertionError("missing/empty required evidence: "+name)


def audit(directory,require_outer=True):
    from hsgd.runtime_gate import JOBS,CHECKS,fixture_case,SEED
    out=Path(directory);checks=[]
    try:
        need(out,["summary.json","launcher.json"])
        summary=json.loads((out/"summary.json").read_text());launcher=json.loads((out/"launcher.json").read_text())
        assert summary["stage"]=="v6-runtime-gate" and summary["formal_training"] is False
        assert summary["tolerances"]==TOLERANCES
        statuses=summary["statuses"]
        assert set(statuses)==set(CHECKS) and statuses and all(v in ("passed","failed","incomplete") for v in statuses.values())
        assert set(summary["jobs"])==set(JOBS)
        complete=all(v=="passed" for v in statuses.values())
        if not complete:
            # Partial evidence never certifies next-stage conditions. Failures != timeouts.
            assert statuses["next_stage_conditions"]!="passed"
            return dict(exit_code=1 if "failed" in statuses.values() else 75,verified=False,statuses=statuses,
                        checks=["explicit partial/failure status; no gate certification"],reason="not all required checks completed")
        assert summary["files"] and len(summary["files"])>=20
        for name,digest in summary["files"].items():
            path=out/name
            assert path.resolve().is_relative_to(out.resolve()) if hasattr(path.resolve(),"is_relative_to") else str(path.resolve()).startswith(str(out.resolve())+"/")
            need(out,[name]);assert sha(path)==digest,"evidence hash mismatch: "+name
        need(out,["run.json","initial.pt","small_final_graph.npz","training_pair.pt","training_pair_input.pt",
                  "training_continuous.pt","training_split.pt","training_recovered.pt",
                  "training_input_1.pt","training_input_2.pt","chain_continuous.pt","chain_split.pt",
                  "chain_resume.pt","chain_interrupted.pt","interrupt_resume.pt","threshold_differences.json"])
        assert [x["job"] for x in launcher["jobs"]]==list(JOBS)
        assert all(x["exit_code"]==0 for x in launcher["jobs"])
        assert 1<=launcher["wall_cap_seconds"]<=570
        run=json.loads((out/"run.json").read_text());binding=run["initial_binding"]
        if require_outer and run.get("bounded_launch"):
            need(out,["outer_launcher.json"])
            outer=json.loads((out/"outer_launcher.json").read_text())
            assert outer["child_exit_code"]==0 and outer["reason"] is None
            assert 30<=outer["hard_cap_seconds"]<=570 and outer["cleanup_grace_seconds"]<=6
        assert binding["code"]==code_hashes() and binding["config"]==config()
        assert binding["backend"]==runtime_binding(run["environment"])
        assert binding["graph_backend"]=="upstream" and binding["T"]==1000 and binding["rounds"]==20
        initial=trusted_load(out/"initial.pt");assert initial["binding"]==binding and initial["completed_updates"]==0
        origin=initial["origin_optimizer_updates"]
        if binding["origin"] is not None:
            info=binding["origin"]
            assert info["old_chain_resume_certified"] is False and origin==100
            assert sha(info["weight_path"])==info["weight_sha256"]
            if "state_path" in info:assert sha(info["state_path"])==info["state_sha256"]
        assert initial["start_time"]>=4
        case=fixture_case() if isinstance(binding["data"],str) and binding["data"].startswith("deterministic") else imdb_case(PACKAGE/"data/IMDB_movie_reference.npz",config())
        for job in JOBS:
            need(out,[job+".json",job+".stdout.log"])
            r=json.loads((out/(job+".json")).read_text())
            assert r==summary["jobs"][job] and r["status"]=="completed" and r["job"]==job
            assert runtime_binding(r["environment"])==binding["backend"]
        checks.append("source/config/data/backend/Python/actual Torch and read-only origin")
        def native(name):
            ck=trusted_load(out/name);assert ck["binding"]==binding
            return ck
        pair=native("training_pair.pt")
        assert pair["local_updates"]==1 and pair["input_sha256"]==sha(out/"training_pair_input.pt")
        a,b=pair["diagnostics"]["legacy"],pair["diagnostics"]["upstream"]
        maxima={k:compare_float(a[k],b[k],k) for k in ("logits","p0","loss")}
        gs=[]
        assert a["gradients"].keys()==b["gradients"].keys()
        for name,v in a["gradients"].items():
            if v is None:assert b["gradients"][name] is None
            else:gs.append(compare_float(v,b["gradients"][name],"gradient"))
        assert gs;maxima["gradient"]=max(gs)
        maxima["updated_parameter"]=max(compare_float(v,pair["upstream"][name],"updated_parameter") for name,v in pair["legacy"].items())
        assert maxima==pair["errors"]
        split=native("training_split.pt");cont=native("training_continuous.pt");recovered=native("training_recovered.pt")
        assert split["completed_updates"]==1 and cont["completed_updates"]==recovered["completed_updates"]==2
        for ck,count in ((split,1),(cont,2),(recovered,2)):
            assert ck["origin_optimizer_updates"]==origin and len(ck["history"])==count
            assert all(int(v["step"])==origin+count for v in ck["optimizer"]["state"].values())
            for i,r in enumerate(ck["history"],1):
                assert r["position"]==i and r["backend"]==r["actual_dispatch"]=="upstream"
                assert r["input_sha256"]==sha(out/("training_input_%d.pt"%i))
        for name,v in cont["model"].items():compare_float(v,recovered["model"][name],"updated_parameter")
        assert tree_error(cont["optimizer"],recovered["optimizer"])<=config()["recovery_atol"]
        compare_rng(cont["rng"],recovered["rng"])
        checks.append("normal train_input, logits/p0/loss/gradients/one update; counts/optimizer/RNG new-process restore")
        with np.load(out/"small_final_graph.npz",allow_pickle=False) as z:
            assert int(z["T"])==8 and np.array_equal(z["types"],toy_case()["space"].types)
            assert graph_stats(toy_case(),z["state"])["schema_simple_undirected"]
        small=summary["jobs"]["small"]
        assert small["completed_reverse_steps"]==8 and small["remaining_time"]==0
        assert small["state_difference_counts"]==[0]*8
        expected_coverage=np.repeat(case["active"].astype(np.uint8),case["space"].counts)
        actualmax=0.
        for i in range(1,5):
            names=["legacy_step_%d.npz"%i,"upstream_step_%d.npz"%i]
            need(out,names)
            with np.load(out/names[0],allow_pickle=False) as a,np.load(out/names[1],allow_pickle=False) as b:
                assert np.array_equal(a["coverage"],expected_coverage) and np.array_equal(b["coverage"],expected_coverage)
                for key in ("uniform","state","coverage"):assert np.array_equal(a[key],b[key]),key
                actualmax=max(actualmax,compare_float(a["q"],b["q"],"reverse"))
                compare_float(a["p0"],b["p0"],"p0")
                for arr in (a,b):
                    query=expected_coverage.astype(bool)
                    assert np.array_equal(arr["state"][query],(arr["uniform"][query]<arr["q"][query]).astype(np.uint8))
        assert json.loads((out/"threshold_differences.json").read_text())==[]
        chain=native("chain_continuous.pt");cs=native("chain_split.pt");ci=native("chain_interrupted.pt")
        for ck,count in ((chain,4),(cs,2),(ci,2),(native("chain_resume.pt"),4),(native("interrupt_resume.pt"),4)):
            assert ck["completed_reverse_steps"]==count and ck["remaining_time"]+count==initial["start_time"]
            assert ck["test_start_time"]==initial["start_time"] and len(ck["history"])==count
        assert np.array_equal(ci["state"],cs["state"]);compare_rng(ci["rng"],cs["rng"])
        for job in ("chain_resume","interrupt_resume"):
            recovered=native(job+".pt")
            assert np.array_equal(chain["state"],recovered["state"]);compare_rng(chain["rng"],recovered["rng"])
            for i in (3,4):
                need(out,[job+"_step_%d.npz"%i])
                with np.load(out/(job+"_step_%d.npz"%i)) as a,np.load(out/("upstream_step_%d.npz"%i)) as b:
                    for key in ("state","uniform","coverage"):assert np.array_equal(a[key],b[key])
                    compare_float(a["q"],b["q"],"reverse")
        checks.append("T8 complete fixture; four-step coverage/state/uniform/RNG; new-process 2+2; partial-step rollback/replay")
        perf=summary["jobs"]["performance"]
        assert perf["rounds"]==20 and len(perf["full_step_timings"])==2
        assert [r["backend"] for r in perf["full_step_timings"]]==["legacy","upstream"]
        assert all(r["covered_slots"]==int(expected_coverage.sum()) and r["step_seconds"]>0 for r in perf["full_step_timings"])
        checks.append("normal-input update and low-interference complete 20-block step timings present")
        return dict(exit_code=0,verified=True,statuses=statuses,checks=checks,
                    max_reverse_error=actualmax,training_errors=maxima,
                    limitations="engineering only; no complete IMDB chain / formal training throughput / quality certification")
    except Exception as exc:
        return dict(exit_code=1,verified=False,checks=checks,error=type(exc).__name__+": "+str(exc))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("directory",type=Path);args=p.parse_args()
    r=audit(args.directory);print(json.dumps(r,indent=2));return r["exit_code"]


if __name__=="__main__":sys.exit(main())
