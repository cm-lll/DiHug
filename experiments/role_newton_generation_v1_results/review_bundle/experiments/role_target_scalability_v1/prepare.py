"""Preparation ONLY: hashes, anonymous targets, flows, exact DP arithmetic/timing.

No denoiser construction, forward, optimizer or diffusion sampling is reachable.
"""
import argparse
from collections import Counter,defaultdict
import importlib.metadata
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"]=""
os.environ["OMP_NUM_THREADS"]="1";os.environ["OPENBLAS_NUM_THREADS"]="1"
import numpy as np
from algorithms import checked,sha,degrees,role_rows,dp_cost,flow_witness,inventory

HERE=Path(__file__).resolve().parent;EXP=HERE.parent
sys.path.insert(0,str(EXP/"role_robust_training_v1"))
from core import load_panel,RoleTarget,torch,canonical,file_manifest
from capacity_readout import allocation_marginals,degree_logpmf,MAX_DP_CELLS
from role_condition.model import capacities


def save(path,x):path.write_text(json.dumps(x,indent=2,allow_nan=False)+"\n")


def main():
    ap=argparse.ArgumentParser()
    for key in ("run","panel","output"):ap.add_argument("--"+key,type=Path,required=True)
    a=ap.parse_args();cfg=json.loads((HERE/"config.json").read_text())
    # Reject protected overlap before any writes. No source or historical output changes.
    out=a.output.resolve()
    protected=[a.run.resolve(),a.panel.resolve(),HERE.resolve(),(EXP/"joint_cn_dual_rounds_v1_1/package/data").resolve()]
    if out.exists() or any(out==p or p in out.parents or out in p.parents for p in protected):raise ValueError("new separate output required")
    start=time.monotonic();cpu=time.process_time();deadline=start+cfg["preparation_algorithm_cap_seconds"]
    def check():
        if time.monotonic()>deadline:raise TimeoutError("bounded preparation algorithm budget")
    a.output.mkdir(parents=True);status="failed";reason=None
    try:
        pinned=EXP/"role_robust_training_6000_v1_results/evidence/learning_run01"
        assert json.loads((pinned/"resolved_config.json").read_text())==json.loads((a.run/"resolved_config.json").read_text())
        config=json.loads((pinned/"resolved_config.json").read_text())
        assert config["updates"]==6000 and config["T"]==64 and config["rounds"]==4
        assert file_manifest()==json.loads((pinned/"source_manifest.json").read_text())
        training=Counter();n=0
        # Pinned public log; no additional training or target inference from outputs.
        for line in (pinned/"updates.jsonl").open():
            r=json.loads(line);n+=1
            assert r["condition"] in cfg["controls"]
            if r["arm"]=="normal2":training[r["seed"],r["condition"]]+=1
        assert n==36000 and set(training.values())=={3000}
        history=dict(baseline_commit=cfg["baseline_commit"],training_log_sha256=sha(pinned/"updates.jsonl"),
            panel_sha256=sha(a.panel/"panel.json"),normal2_condition_counts=[dict(seed=k[0],condition=k[1],updates=v) for k,v in sorted(training.items())],
            seen_targets_only=list(cfg["controls"]),new_targets_used_in_training=False)
        h=json.loads((pinned/"weight_hashes.json").read_text());weights=[]
        for seed in cfg["train_seeds"]:
            check();rel=f"seedjobs/seed{seed}/seed{seed}/normal2_final_6000.pt";cp=f"seedjobs/seed{seed}/seed{seed}/pair_u006000.pt"
            checked(a.run/rel,h[rel]);checked(a.run/cp,h[cp])
            state=torch.load(a.run/rel,map_location="cpu",weights_only=True)
            ck=torch.load(a.run/cp,map_location="cpu",weights_only=False)
            assert ck["completed"]=={"normal2":6000,"robust":6000} and ck["seed"]==seed
            assert ck["binding"]["config_sha256"]==sha(EXP/"role_robust_training_v1/config.json")
            assert ck["binding"]["source_manifest"]==json.loads((pinned/"source_manifest.json").read_text())
            assert ck["binding"]["inputs"]==json.loads((pinned/"input_bindings.json").read_text())
            for fp,expected in ck["binding"]["inputs"].items():checked(fp,expected)
            assert ck["binding"]["environment"]["packages"]=={k:importlib.metadata.version(k) for k in ck["binding"]["environment"]["packages"]}
            assert state.keys()==ck["models"]["normal2"].keys()
            assert all(torch.equal(v,ck["models"]["normal2"][k]) for k,v in state.items())
            assert all(torch.isfinite(v).all() for v in state.values())
            layers=sorted({int(k.split(".")[2]) for k in state if k.startswith("backbone.tf_layers.")})
            assert layers==list(range(8)) and tuple(state["backbone.lin_in_X.weight"].shape)[0]==256
            weights.append(dict(seed=seed,arm="normal2",updates=6000,path=str(a.run/rel),sha256=h[rel],paired_checkpoint_sha256=h[cp],
                tensors=len(state),all_tensors_match_complete_checkpoint=True,all_finite=True,transformer_layers=len(layers),
                parameter_count_from_archived_initialization=json.loads((a.run/f"seedjobs/seed{seed}/seed{seed}/initialization.json").read_text())["parameters"],
                checkpoint_binding=ck["binding"],
                model_instantiated=False,model_forward_calls=0))
            del ck,state
        save(a.output/"weights.json",dict(config_sha256=sha(pinned/"resolved_config.json"),source_manifest_sha256=sha(pinned/"source_manifest.json"),
            full_resolved_config=config,weights=weights,history=history,source_files_exact=True))
        records,seen=load_panel(a.panel);base=seen["balanced"]
        assert all(base.roles[t]==seen["extreme"].roles[t] for t in range(3))
        case=next(r["case"] for r in records if r["condition"]=="balanced")
        types=case["space"].types;old=case["clean"].T;fam=case["space"].encode(old.T)[0];R=case["n_families"]
        d=degrees(types,old,fam,case["space"].pairs);movies=np.flatnonzero(types==2);genres=np.flatnonzero(types==3);relation=case["space"].pairs.index((2,3))
        assert np.bincount(d[movies,relation],minlength=3).tolist()==[0,12,6]
        target_records=[];proofs=[];timings=[];bench_start=time.monotonic()
        proof_dir=a.output/"proof_only";proof_dir.mkdir()
        for name,vector in {**cfg["controls"],**cfg["unseen"]}.items():
            check();rows=list(base.roles);cnt=Counter(vector)
            rows[3]=tuple((tuple(v if r==relation else 0 for r in range(R)),n) for v,n in sorted(cnt.items()))
            target=RoleTarget(tuple(rows),base.type_counts,base.schema)
            assert all(target.roles[t]==base.roles[t] for t in range(3))
            if name in cfg["controls"]:assert target==seen[name]
            else:assert target.fingerprint not in {v.fingerprint for v in seen.values()}
            pairs,proof=flow_witness(d[movies,relation],vector)
            fresh=np.sort(np.column_stack((movies[pairs[:,0]],genres[pairs[:,1]])),axis=1)
            witness=np.concatenate((old[fam!=relation],fresh),axis=0);wf=np.r_[fam[fam!=relation],np.full(len(fresh),relation)]
            wd=degrees(types,witness,wf,case["space"].pairs)
            assert role_rows(wd,types)==target.roles
            assert np.array_equal(np.bincount(wf,minlength=R),case["counts"])
            path=proof_dir/(name+".npz");np.savez_compressed(path,types=types,edges=witness,relations=wf+1,schema=np.asarray(target.schema))
            proofs.append(dict(target=name,**proof,proof_file=str(path.relative_to(a.output)),sha256=sha(path),
                all_other_node_relation_degrees_preserved=bool(np.array_equal(wd[types!=3],d[types!=3])),
                full_joint_target_verified=True,physical_edges=len(witness),role_fingerprint=target.fingerprint,
                forbidden_for_generation=True))
            target_records.append(dict(name=name,seen_in_training=name in cfg["controls"],Genre_degrees=vector,
                fingerprint=target.fingerprint,anonymous_target=target.as_dict(),DP=[dp_cost([n for v,n in rows]) for rows in target.roles],
                degree_ge8=sum(v>=8 for v in vector),degree_ge12=sum(v>=12 for v in vector),degree_le1=sum(v<=1 for v in vector),degree_zero=vector.count(0)))
            for typ,rows in enumerate(target.roles):
                n=target.type_counts[typ];k=len(rows);c=np.array([c for v,c in rows])
                ll=-((np.arange(n)[:,None]+1)/(n+1)-(np.arange(k)[None,:]+1)/(k+1))**2
                duration=[]
                for repeat in range(cfg["algorithm_benchmark_repetitions"]):
                    tick=time.perf_counter();attention,info=allocation_marginals(ll,c);duration.append(time.perf_counter()-tick)
                    np.testing.assert_allclose(attention.sum(0),c,rtol=0,atol=2e-9)
                timings.append(dict(target=name,type=typ,role_kinds=k,synthetic_log_likelihood=True,model_probabilities=False,
                    first_seconds=duration[0],warm_median_seconds=float(np.median(duration[1:])),repetitions=len(duration),DP=info))
        save(a.output/"targets.json",dict(frozen_before_generation=True,training_history=history,targets=target_records,
            generation_input_allowlist=["normal2@6000", "anonymous_target", "original_schema_prior_schedule", "paired_sampling_seed"],
            proof_graphs_forbidden=True,model_calls=0))
        save(a.output/"feasibility.json",dict(proofs=proofs,all_feasible=True,not_scientific_samples=True))
        save(a.output/"small_algorithm_timings.json",dict(records=timings,total_wall_seconds=time.monotonic()-bench_start,
            likelihood="fixed synthetic finite LL, loop-cost probe only; not model predictions or generation quality"))
        datasets={};origin={};root=EXP/"joint_cn_dual_rounds_v1_1/package"
        historic=json.loads((root/"data/historical_inventory.json").read_text())
        archived=json.loads((root/"data/verified_inventory.json").read_text())
        for name in ("IMDB","PubMed"):
            check();dcfg=json.loads((root/"configs"/(name+".json")).read_text());path=root/dcfg["data"]["file"];checked(path,dcfg["data"]["sha256"])
            with np.load(path,allow_pickle=False) as z:inv=inventory(z["types"],z["edges"],z["relations"]-1,z["schema"],dcfg["data"]["type_names"])
            assert inv["nodes"]==dcfg["data"]["nodes"] and inv["physical_edges"]==dcfg["data"]["edges"]
            assert inv["reference_wedges"]==archived[name]["CN_forward_noise_risk"]["reference_wedges"]
            inv["source_path"]=str(path);inv["sha256"]=dcfg["data"]["sha256"]
            inv["CN_forward_noise_risk_historical"]= {k:v for k,v in archived[name]["CN_forward_noise_risk"].items() if k!="forward_curve"}
            inv["T_full_scale_planning_only"]=1000;inv["query_rounds_full_scale"]=20
            inv["backbone_calls_per_chain"]=20000;inv["role_reads_per_chain"]=1000
            inv["state_uint8_bytes"]=inv["all_slots"];inv["one_float64_full_probability_bytes"]=8*inv["all_slots"]
            inv["NN_new_timing_executed"]=False;inv["exact_pipeline_can_complete"]=False
            inv["first_type_DP_guard_failure"]=next(t["name"] for t in inv["types"] if not t["DP"]["guard_passed"])
            # Actual source guard called only with a small likelihood matrix; never enumerate DP states.
            for t in inv["types"]:
                c=np.array([x["frequency"] for x in t["roles"]]);ll=np.zeros((t["nodes"],len(c)))
                try:allocation_marginals(ll,c)
                except RuntimeError as exc:
                    assert "capacity DP resource limit" in str(exc);t["original_guard_checked"]=str(exc)
                else:raise AssertionError("unexpected full data DP acceptance")
            key="IMDB_movie" if name=="IMDB" else "PubMed_baseline";rec=historic[key]
            provenance=[]
            for fp,expected in rec["input_hashes"].items():
                pp=Path(fp);actual=sha(pp) if pp.is_file() else None
                provenance.append(dict(path=fp,expected_historical_sha256=expected,readable=pp.is_file(),sha256=actual,matches=actual==expected if actual else None))
            inv["provenance"]=dict(source=rec["source"],preprocessed_version=rec["version"],
                historical_inventory_sha256=sha(root/"data/historical_inventory.json"),verified_inventory_sha256=sha(root/"data/verified_inventory.json"),payloads=provenance)
            datasets[name]=inv
        save(a.output/"large_graph_inventory.json",datasets)
        gen=json.loads((pinned/"generation.json").read_text())
        old=[r for r in gen if r["arm"]=="normal2" and r["checkpoint_update"]==6000]
        costs=[r["seconds"] for r in old];p90=float(np.quantile(costs,.9));med=float(np.median(costs))
        old_soft=max(sum(x["warm_median_seconds"] for x in timings if x["target"]==name) for name in cfg["controls"])
        new_soft=max(sum(x["warm_median_seconds"] for x in timings if x["target"]==name) for name in cfg["unseen"])
        extra=max(0,new_soft-old_soft)*64
        forecast=dict(historical_same_weights_controls=48,historical_chain_median_seconds=med,historical_chain_p90_seconds=p90,
            historical_generation_sha256=sha(pinned/"generation.json"),planned_chains=120,new_condition_chains=72,controls=48,
            exact_DP_increment_estimate_per_chain_seconds=extra,estimate_scope="CPU; historical full chain timing plus synthetic-LL exact DP increment; no new model timings",
            simple_serial_CPU_seconds=120*(p90+extra),three_seed_parallel_simple_wall_seconds=40*(p90+extra),
            suggested_budget_seconds=2700,budget_formula="2*simple_three_seed_wall + 300 seconds load/evaluation/audit/IO; cap proposed2700, not activated",
            budget_safety_estimate_seconds=2*40*(p90+extra)+300,
            unknowns=["unseen-condition reverse-state density/CN and numerical behavior", "cold/warm model/cache effects", "hardware load"],
            optional_model_short_test=dict(authorized=False,cap_seconds=120,device="CPU",fixed_old_state="frozen holdout_0_t8.npz",
                work="3 final normal2 weights x5 targets x complete4 query blocks=60 eval/no_grad forwards; no state sampling or update",
                starts_no_scientific_chain=True),big_graph_estimate="exact DP resource guard rejects; no finite end-to-end runtime or GPU-memory claim")
        assert forecast["budget_safety_estimate_seconds"]<2700
        save(a.output/"cost_plan.json",forecast)
        # End-of-task protected payload checks.
        for w in weights:checked(w["path"],w["sha256"])
        for name,inv in datasets.items():checked(inv["source_path"],inv["sha256"])
        assert not torch.cuda.is_initialized();status="completed"
    except BaseException as exc:reason=repr(exc);raise
    finally:
        save(a.output/"receipt.json",dict(status=status,reason=reason,wall_seconds=time.monotonic()-start,CPU_seconds=time.process_time()-cpu,
            peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,model_instantiations=0,model_forward_calls=0,
            optimizer_updates=0,scientific_chains=0,GPU=0,proof_witnesses_not_samples=True,
            environment={k:importlib.metadata.version(k) for k in ("torch","torch-geometric","numpy","scipy","networkx","pytest")},python=sys.version,
            no_remote_data_access=True,algorithm_cap_seconds=cfg["preparation_algorithm_cap_seconds"]))

if __name__=="__main__":main()
