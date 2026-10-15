"""CPU-only post-run evidence audit; never invokes a model or sampler."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
from hsgd.bw_preflight.common import PACKAGE,sha,trusted_load,code_hashes,config
from hsgd.learning_budget import signature,model_fingerprint
from hsgd.learning import verify_training_gate
from hsgd.learning_control import validate as validate_control
from hsgd.learning_mode import require as require_mode,same_platform


def audit(out):
    out=Path(out);checks=[]
    try:
        need=lambda name:(_ for _ in ()).throw(RuntimeError("missing/empty evidence: "+name)) if not (out/name).is_file() or not (out/name).stat().st_size else None
        for n in ("freeze.json","run_control.json","latest.json","summary.json","preflight/training_gate.json","validation_inputs/manifest.json"):need(n)
        if not (out/"updates.jsonl").is_file():raise RuntimeError("missing update ledger")
        gate=verify_training_gate(out/"preflight");checks.append("independent normal-input numeric/gradient/update/new-process preflight")
        freeze=json.loads((out/"freeze.json").read_text());binding=freeze["binding"]
        if binding["code"]!=code_hashes() or binding["config"]!=config() or binding["graph_backend"]!="upstream":raise RuntimeError("source/config/backend binding")
        control=json.loads((out/"run_control.json").read_text());validate_control(control)
        if control!=binding["run_control"]:raise RuntimeError("run control binding")
        require_mode(binding["backend"],False)
        same_platform(gate["binding"]["backend"],binding["backend"])
        source=PACKAGE/"data/IMDB_movie_reference.npz"
        if not control["engineering_fixture"]:
            if sha(source)!=binding["data"] or sha(PACKAGE/"data/IMDB_preprocessing_provenance.json")!=binding["data_provenance_sha256"]:raise RuntimeError("data provenance")
        latest=json.loads((out/"latest.json").read_text());path=out/latest["path"]
        if not path.resolve().is_relative_to(out.resolve()) or sha(path)!=latest["sha256"]:raise RuntimeError("checkpoint path/hash")
        ck=trusted_load(path);s=json.loads((out/"summary.json").read_text())
        if ck["model_sha256"]!=model_fingerprint(ck["model"]):raise RuntimeError("checkpoint weight fingerprint")
        if ck["binding"]!=binding or ck["completed_updates"]!=latest["completed_updates"]:raise RuntimeError("checkpoint counts/binding")
        n=ck["completed_updates"]
        from hsgd.cn_optimizer import verify_steps
        verify_steps(ck)
        if signature({k:v for k,v in ck["optimizer"]["param_groups"][0].items() if k!="params"})!=signature(freeze["actual_dimensions"]["optimizer"]):raise RuntimeError("optimizer settings")
        if not all(torch.isfinite(v).all() for v in ck["model"].values()):raise RuntimeError("nonfinite checkpoint")
        rows=[json.loads(x) for x in (out/"updates.jsonl").read_text().splitlines() if x.strip()]
        if len(rows)<n or rows[:n]!=ck["history"]:raise RuntimeError("durable ledger mismatch")
        for i,r in enumerate(rows,1):
            require_mode(r["execution_mode"],False)
            if r["update"]!=i or r["backend"]!=r["actual_dispatch"] or r["backend"]!="upstream" or r["skipped"]:raise RuntimeError("update dispatch/count ledger")
            if not 1<=r["time"]<=1000 or np.sum(r["family_label_counts"])!=r["query_slots"]:raise RuntimeError("time/query label ledger")
        checks.append("bound checkpoint/Adam/updates/RNG presence and explicit stop policy; no time-budget dependency")
        if set(ck["rng"])!={"generator","python","numpy","torch","accelerator"}:raise RuntimeError("incomplete RNG")
        meta=json.loads((out/"validation_inputs/manifest.json").read_text())
        if sha(out/"validation_inputs/manifest.json")!=freeze["validation_input_manifest_sha256"]:raise RuntimeError("validation manifest binding")
        for name,digest in meta["files"].items():
            if sha(out/"validation_inputs"/name)!=digest:raise RuntimeError("frozen input binding")
        records=[]
        cached_bindings=[]
        for file in sorted((out/"validation").glob("*/validation.json")):
            r=json.loads(file.read_text());records.append(r)
            if r["status"]!="completed" or r["input_manifest_sha256"]!=freeze["validation_input_manifest_sha256"]:raise RuntimeError("validation source")
            require_mode(r["execution_mode"],False)
            if [x["time"] for x in r["times"]]!=[1,250,500,750,1000]:raise RuntimeError("validation time coverage")
            for t in r["times"]:
                c=np.array(t["counts"]);loss=np.array(t["loss_sums"])
                if c.sum()!=sum(freeze["reference"]["slots"]) or not np.array_equal(c[:,1],freeze["reference"]["relations"]) or t["query_blocks"]!=20:raise RuntimeError("full validation population")
                if not np.isclose(t["pooled_ce"],loss.sum()/c.sum(),rtol=1e-12,atol=1e-12):raise RuntimeError("validation reduction")
                with np.load(file.parent/("expected_degree_t%04d.npz"%t["time"])) as z:
                    if not np.allclose(z["expected_degree"].sum(0),2*np.array(t["expected_family_edges"]),rtol=1e-12,atol=1e-10):raise RuntimeError("degree/edge mass")
            for name,digest in r["prediction_files"].items():
                if sha(file.parent/name)!=digest:raise RuntimeError("cached prediction hash")
                with np.load(file.parent/name) as z:
                    input_name=str(z["input_id"])+".npz"
                    if str(z["input_sha256"])!=meta["files"][input_name] or int(z["completed_updates"])!=r["completed_updates"]:raise RuntimeError("prediction/input/stage binding")
                    if str(z["model_sha256"])!=r["model_sha256"]:raise RuntimeError("prediction/weight binding")
                    with np.load(out/"validation_inputs"/input_name) as inp:
                        if z["p0"].shape!=inp["labels"].shape or z["p0"].dtype!=np.float32:raise RuntimeError("prediction shape/dtype")
            if r["prediction_files"]:cached_bindings.append((r["completed_updates"],r["model_sha256"]))
        saved_bindings=set()
        for p in (out/"checkpoints").glob("*.pt"):
            saved=trusted_load(p)
            if saved["binding"]!=binding or saved["model_sha256"]!=model_fingerprint(saved["model"]):raise RuntimeError("retained checkpoint binding")
            saved_bindings.add((saved["completed_updates"],saved["model_sha256"]))
            del saved
        if not set(cached_bindings).issubset(saved_bindings):raise RuntimeError("cached probabilities lack corresponding retained weights")
        checks.append("all recorded full20 validation populations/reductions/degree mass and cached input IDs")
        stages=set(r["stage"] for r in records if len(r["prediction_files"])==5)
        if s["status"]=="failed":return dict(exit_code=1,verified=False,checks=checks,reason=s["failure"])
        required={"initial","final"}
        if ck["periodic_prediction_saved"]:required.add("first_periodic")
        if not required.issubset(stages):return dict(exit_code=75,verified=False,checks=checks,reason="prediction stages incomplete",observed_stages=sorted(stages))
        if s["final_checkpoint"]!=latest["path"] or s["completed_updates"]!=n:raise RuntimeError("final weights/count mismatch")
        for p in out.glob("resume_loaded_*.json"):
            loaded=json.loads(p.read_text());require_mode(loaded["execution_mode"],False)
            if not all(loaded.get(k) is True for k in ("loaded_weights_exact","loaded_optimizer_exact","loaded_RNG_exact")):
                raise RuntimeError("resume did not verify exact loaded state")
            if loaded["post_resume_trajectory_identity_required"] is not False:raise RuntimeError("ordinary replay contract changed")
        checks.append("ordinary-mode state-load verification records; no counterfactual trajectory equality claimed")
        return dict(exit_code=0 if s["status"] in ("stopped_manual","stopped_limit") else 75,verified=True,checks=checks,
                    status=s["status"],completed_updates=n,final_checkpoint=latest["path"],
                    generation_quality="not tested",converged="not certified",scope="CPU consistency audit, not model replay")
    except Exception as exc:return dict(exit_code=1,verified=False,checks=checks,error=type(exc).__name__+": "+str(exc))


def main():
    p=argparse.ArgumentParser();p.add_argument("directory",type=Path)
    p.add_argument("--arm",choices=("baseline","cn"),required=True);args=p.parse_args()
    from hsgd.bw_preflight.common import set_arm
    set_arm(args.arm)
    r=audit(args.directory);print(json.dumps(r,indent=2));return r["exit_code"]


if __name__=="__main__":sys.exit(main())
