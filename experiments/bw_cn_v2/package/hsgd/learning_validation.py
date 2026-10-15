"""Full-coverage same-reference p0 validation. No reverse chain / graph sampling.

CE uses FP32 BCE per slot accumulated in float64. Full legal-slot denominator
includes skipped exact-none families (loss0). No edge/class balancing change.
"""
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from hsgd.bw_preflight.common import (json_save,sha,plan_for,schedule,rng_save,rng_restore,
                                    check_prob_device,sync)
from hsgd.noising import global_noisy_edges
from hsgd.computational import FrozenTypedContext
from hsgd.runtime import predict
from hsgd.learning_budget import model_fingerprint
from hsgd.learning_mode import state as mode_state


def reference_degree(case):
    degree=np.zeros((len(case["space"].types),case["n_families"]),dtype=np.float64)
    f=case["space"].encode(case["clean"])[0]
    for r in range(case["n_families"]):degree[:,r]=np.bincount(case["clean"][:,f==r].ravel(),minlength=len(degree))
    return degree


def baseline(case):
    pi=torch.tensor(case["prior"],dtype=torch.float32)
    pos=torch.nn.functional.binary_cross_entropy(pi,torch.ones_like(pi),reduction="none").double().numpy()
    neg=torch.nn.functional.binary_cross_entropy(pi,torch.zeros_like(pi),reduction="none").double().numpy()
    m=case["counts"].astype(np.int64);n=case["space"].counts.astype(np.int64)
    return dict(prior=case["prior"].tolist(),prior_compute_dtype="FP32 (same BCE formula); densities recorded float64",
        family_positive_sum=(m*pos).tolist(),family_negative_sum=((n-m)*neg).tolist(),
        family_sum=(m*pos+(n-m)*neg).tolist(),positive_count=m.tolist(),negative_count=(n-m).tolist(),
        pooled_ce=float((m*pos+(n-m)*neg).sum()/case["space"].total),
        family_ce=np.divide(m*pos+(n-m)*neg,n,out=np.zeros(len(n)),where=n>0).tolist())


def freeze_inputs(case,cfg,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    sizes=[];files={};T,R=cfg["T"],cfg["rounds"]
    for t in cfg["validation_times"]:
        seed=[cfg["validation_seed"],t,91]
        ei,ef,stats=global_noisy_edges(case["space"],case["clean"],case["prior"],schedule(T)[t],np.random.default_rng(seed))
        ids,f,qe=plan_for(case,[cfg["validation_seed"],t,92],R).block(0)
        labels=np.isin(ids,case["clean_ids"]).astype(np.uint8)
        name="val_t%04d_block00.npz"%t
        np.savez_compressed(out/name,time=np.array(t),edges=ei,edge_families=ef,ids=ids,
                            families=f,query=qe,labels=labels)
        files[name]=sha(out/name);sizes.append(dict(id=name[:-4],t=t,block=0,slots=len(ids),noisy_edges=len(ef),noise_stats=stats))
    cache_slots=sum(x["slots"] for x in sizes)
    # ZIP can be incompressible: estimate uncompressed bytes, not a compression claim.
    estimate=dict(predictions_per_snapshot_upper_bytes=cache_slots*4+4096*len(sizes),
                  three_snapshot_upper_bytes=cache_slots*4*3+4096*len(sizes)*3,
                  fixed_inputs_actual_bytes=sum((out/n).stat().st_size for n in files),
                  fixed_inputs_worst_uncompressed_bytes=len(sizes)*int(np.sum(case["space"].counts[case["active"]]))*24+cache_slots*33+4096*len(sizes),
                  note="only 5 predetermined block0 inputs; initial/first-periodic/final p0 FP32; no full per-update probabilities")
    meta=dict(seed=cfg["validation_seed"],times=cfg["validation_times"],rounds=R,
        noise_seed_rule="[validation_seed,t,91]",plan_seed_rule="[validation_seed,t,92]",files=files,
        identifiers=sizes,disk_estimate=estimate,
        meaning="one fixed forward-noised whole reference per time; all20 queries share it; same-reference validation only",
        zero_support="inactive exact-none family CE0 included in full schema denominator",
        tail_definition="top ceil(.10*n_type) reference-degree nodes, and sorted expected-degree tail; ties broken by ID only for diagnostics",
        loss="FP32 BCE per physical upper-triangle slot; float64 accumulation; mean across fixed time points, each S denominator")
    json_save(out/"manifest.json",meta);return meta


def calibration(case,degree):
    ref=reference_degree(case);rows=[]
    for f,(a,b) in enumerate(case["space"].pairs):
        for ty in sorted(set((a,b))):
            nodes=np.flatnonzero(case["space"].types==ty);x=degree[nodes,f];y=ref[nodes,f]
            k=max(1,math.ceil(.1*len(nodes)))
            order=np.lexsort((nodes,-y))[:k]
            rows.append(dict(family=f,type=int(ty),nodes=len(nodes),tail_nodes=k,
                expected_sum=float(x.sum()),reference_sum=float(y.sum()),expected_mean=float(x.mean()),
                expected_p90=float(np.quantile(x,.9)),expected_max=float(x.max()),
                reference_p90=float(np.quantile(y,.9)),reference_max=float(y.max()),
                sorted_distribution_l1_normalized=float(np.abs(np.sort(x)-np.sort(y)).sum()/max(1.,y.sum())),
                reference_top10pct_predicted_mass=float(x[order].sum()),reference_top10pct_reference_mass=float(y[order].sum()),
                predicted_sorted_top10pct_mass=float(np.sort(x)[-k:].sum())))
    return rows


def validate(model,case,cfg,input_dir,output_dir,training_rng,device,stage,completed,check_time=None,event=None,save_probabilities=False):
    input_dir,output_dir=Path(input_dir),Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=False)
    meta=json.loads((input_dir/"manifest.json").read_text())
    for name,digest in meta["files"].items():
        if sha(input_dir/name)!=digest:raise RuntimeError("frozen validation input changed: "+name)
    if meta["times"]!=cfg["validation_times"] or meta["rounds"]!=cfg["rounds"]:raise RuntimeError("validation protocol changed")
    saved=rng_save(training_rng,device);was_training=model.training;model.eval()
    started=time.perf_counter();cpu=time.process_time();rows=[];pred_files={}
    model_sha256=model_fingerprint(model.state_dict())
    execution_mode=mode_state()
    try:
        with torch.no_grad():
            for t in cfg["validation_times"]:
                if check_time:check_time()
                name="val_t%04d_block00.npz"%t
                with np.load(input_dir/name,allow_pickle=False) as z:
                    ei,ef=z["edges"],z["edge_families"]
                    first={key:z[key].copy() for key in ("ids","families","query","labels")}
                plan=plan_for(case,[cfg["validation_seed"],t,92],cfg["rounds"])
                token=("same-reference-validation",stage,t)
                ctx=FrozenTypedContext(model,case["space"].types,ei,ef,token)
                visited=np.zeros(case["space"].total,dtype=np.uint8)
                degree=np.zeros((len(case["space"].types),case["n_families"]),dtype=np.float64)
                sums=np.zeros((case["n_families"],2));counts=np.zeros_like(sums,dtype=np.int64)
                edges_expected=np.zeros(case["n_families"])
                try:
                    for block in range(cfg["rounds"]):
                        if mode_state()!=execution_mode:raise RuntimeError("validation execution mode changed")
                        if check_time:check_time()
                        ids,f,qe=plan.block(block)
                        if not len(ids):continue
                        if visited[ids].any():raise AssertionError("validation duplicate coverage")
                        labels=np.isin(ids,case["clean_ids"]).astype(np.float32)
                        if block==0:
                            for key,value in (("ids",ids),("families",f),("query",qe),("labels",labels)):
                                if not np.array_equal(value,first[key]):raise AssertionError("frozen validation label/query binding")
                        p=predict(model,case["space"].types,ei,ef,qe,f,t/cfg["T"],"upstream",ctx,token)
                        check_prob_device(p,f,case,device_prior=model.prior)
                        ce=torch.nn.functional.binary_cross_entropy(p,torch.as_tensor(labels,device=device),reduction="none")
                        if not torch.isfinite(ce).all():raise RuntimeError("nonfinite validation CE")
                        pp=p.detach().cpu().numpy();loss=ce.detach().cpu().numpy().astype(np.float64)
                        for label in (0,1):
                            mask=labels==label
                            sums[:,label]+=np.bincount(f[mask],weights=loss[mask],minlength=case["n_families"])
                            counts[:,label]+=np.bincount(f[mask],minlength=case["n_families"])
                        edges_expected+=np.bincount(f,weights=pp.astype(np.float64),minlength=case["n_families"])
                        np.add.at(degree,(qe[0],f),pp);np.add.at(degree,(qe[1],f),pp)
                        visited[ids]+=1
                        if block==0 and save_probabilities:
                            file=name[:-4]+"_p0.npz"
                            np.savez_compressed(output_dir/file,p0=pp,time=np.array(t),completed_updates=np.array(completed),
                                                input_sha256=np.array(meta["files"][name]),input_id=np.array(name[:-4]),
                                                model_sha256=np.array(model_sha256))
                            pred_files[file]=sha(output_dir/file)
                        if event:event("validation_block",stage=stage,time=t,block=block+1,query_slots=len(ids))
                    expected=np.repeat(case["active"].astype(np.uint8),case["space"].counts)
                    if not np.array_equal(visited,expected):raise AssertionError("validation incomplete coverage")
                    counts[~case["active"],0]=case["space"].counts[~case["active"]]
                    if not np.array_equal(counts[:,1],case["counts"]) or not np.array_equal(counts.sum(1),case["space"].counts):raise AssertionError("validation population mismatch")
                    if not np.allclose(degree.sum(0),2*edges_expected,rtol=1e-12,atol=1e-10):raise AssertionError("expected degree/edge mass mismatch")
                    np.savez_compressed(output_dir/("expected_degree_t%04d.npz"%t),expected_degree=degree)
                    rows.append(dict(time=t,query_blocks=cfg["rounds"],covered_active_slots=int(visited.sum()),
                        physical_slots=case["space"].total,counts=counts.tolist(),loss_sums=sums.tolist(),
                        pooled_ce=float(sums.sum()/case["space"].total),
                        family_ce=np.divide(sums.sum(1),counts.sum(1),out=np.zeros(case["n_families"]),where=counts.sum(1)>0).tolist(),
                        positive_ce=float(sums[:,1].sum()/counts[:,1].sum()) if counts[:,1].sum() else None,
                        negative_ce=float(sums[:,0].sum()/counts[:,0].sum()) if counts[:,0].sum() else None,
                        expected_edges=float(edges_expected.sum()),expected_family_edges=edges_expected.tolist(),
                        reference_family_edges=case["counts"].tolist(),expected_degree_diagnostics=calibration(case,degree),
                        probability_object="p0 clean prediction on fixed forward-noised Xt, NOT generated reverse q"))
                finally:ctx.close()
        sync(device)
        b=baseline(case);total_sums=np.sum([r["loss_sums"] for r in rows],axis=0);total_counts=np.sum([r["counts"] for r in rows],axis=0)
        result=dict(stage=stage,status="completed",completed_updates=completed,model_sha256=model_sha256,times=rows,baseline=b,
            execution_mode=execution_mode,
            pooled_ce=float(total_sums.sum()/total_counts.sum()),
            family_ce=np.divide(total_sums.sum(1),total_counts.sum(1),out=np.zeros(case["n_families"]),where=total_counts.sum(1)>0).tolist(),
            positive_ce=float(total_sums[:,1].sum()/total_counts[:,1].sum()) if total_counts[:,1].sum() else None,
            negative_ce=float(total_sums[:,0].sum()/total_counts[:,0].sum()) if total_counts[:,0].sum() else None,
            loss_sums=total_sums.tolist(),counts=total_counts.tolist(),input_manifest_sha256=sha(input_dir/"manifest.json"),
            prediction_files=pred_files,wall_seconds=time.perf_counter()-started,cpu_seconds=time.process_time()-cpu,
            interpretation="same-reference denoising only, not unseen-graph or complete generation quality")
        json_save(output_dir/"validation.json",result);return result
    finally:
        model.train(was_training);rng_restore(saved,training_rng,device)
