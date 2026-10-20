import os
os.environ.update(OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",CUDA_VISIBLE_DEVICES="")
import argparse
import gc
import hashlib
import json
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import numpy as np
import torch
from core import *
from hsgd.runtime import StepPredictor
from role_condition.metrics import graph,compare
from panel import response as fixed_response

def append(path,row):
    with path.open("a") as f:f.write(json.dumps(row,allow_nan=False)+"\n")

def check_source(source,panel,cfg):
    archived=EXP/"role_specialization_v1_results/evidence/learning_run01"
    assert fingerprint(source/"resolved_config.json")==fingerprint(archived/"resolved_config.json")
    assert fingerprint(source/"weight_hashes.json")==fingerprint(archived/"weight_hashes.json")
    for name in ("source_manifest.json","environment.json"):
        assert fingerprint(source/name)==fingerprint(archived/name),name
    assert fingerprint(panel/"panel.json")=="d69e554a3ec43bea1b039bc25f45bf9f4b49e1485dae4b2504fcbc7ce1df9bf0"
    old=json.loads((source/"resolved_config.json").read_text())
    for key in ("model","optimizer","T","rounds","train_seeds","validation_seed","sampling_seeds"):
        assert cfg[key]==old[key],key
    for name,h in json.loads((source/"source_manifest.json").read_text()).items():assert fingerprint(EXP/name)==h,name
    env=c.environment("cpu");before=json.loads((source/"environment.json").read_text())
    for key in ("python_version","packages","torch_runtime_version","float32_matmul_precision","matmul_allow_tf32","deterministic_algorithms","torch_threads"):
        assert env[key]==before[key],key
    hashes={str(p):fingerprint(p) for p in (source/"resolved_config.json",source/"weight_hashes.json",source/"environment.json",panel/"panel.json")}
    expected=json.loads((archived/"weight_hashes.json").read_text())
    for seed in cfg["train_seeds"]:
        p=source/f"seed{seed}/capacity_C_initial.pt";h=fingerprint(p)
        assert h==expected[str(p.relative_to(source))];hashes[str(p)]=h
    return hashes

def freeze(a,cfg,records,targets,check):
    holdout=make_holdout(records,cfg,check)
    save(a.output/"holdout.json",dict(seed=cfg["holdout_seed"],schema=records[0]["case"]["space"].pairs,
        type_names=NAMES,type_counts=[4,6,18,6],nodes=34,
        graphs=[dict(id=r["id"],condition=r["condition"],split=r["split"],types=r["case"]["space"].types.tolist(),
                     edges=r["case"]["clean"].T.tolist()) for r in holdout],
        anonymous_targets={k:v.as_dict() for k,v in targets.items()},
        split_evidence="typed-isomorphism exclusion of all twelve original topologies and earlier heldouts; same anonymous joint roles",
        generation="fixed DMA skeleton plus degree-preserving MG swaps; no learned graph sampling"))
    pressure=[];cases=[]
    for i,rec in enumerate(holdout):
        cc=rec["case"]
        for t in cfg["holdout_times"]:
            rng=np.random.default_rng([cfg["holdout_noise_seed"],i,t])
            inp=c.train_input(cc,cfg["T"],cfg["rounds"],cfg["holdout_query_seed"],i*64+t,rng,force_t=t)
            arrays={name:inp[name] for name in ("ids","f","query","labels","edges","edge_families")}
            arrays.update(types=cc["space"].types,clean_edges=cc["clean"],prior=cc["prior"],
                          schema=np.asarray(cc["space"].pairs),t=np.array(t),round_id=np.array(inp["round_id"]),ce_weight=np.array(inp["ce_weight"]))
            name=f"holdout_{i}_t{t}.npz";np.savez_compressed(a.output/name,**arrays)
            cases.append(dict(id=rec["id"],condition=rec["condition"],i=i,t=t,file=name,sha256=fingerprint(a.output/name)))
            for j,variant in enumerate(cfg["holdout_variants"][1:]):
                direction="add" if variant.startswith("add") else "delete";k=int(variant[-1])
                changed,mask,info=perturb(cc,inp,np.random.default_rng([cfg["holdout_edits_seed"],i,t,j]),cfg["perturbation"],direction,k)
                np.savez_compressed(a.output/f"holdout_{i}_t{t}_{variant}.npz",edges=changed["edges"],edge_families=changed["edge_families"],edit=mask)
                pressure.append(dict(id=rec["id"],condition=rec["condition"],t=t,variant=variant,**info))
    save(a.output/"pressure.json",pressure);save(a.output/"holdout_cases.json",cases)
    save(a.output/"resolved_config.json",cfg);save(a.output/"source_manifest.json",file_manifest())
    save(a.output/"environment.json",c.environment("cpu"))
    try:revision=subprocess.check_output(["git","rev-parse","HEAD"],cwd=HERE,text=True,stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:revision=None
    save(a.output/"revision.json",dict(commit=revision,source_manifest_is_authoritative=True,baseline=cfg["model_baseline_commit"]))
    manifest={p.name:fingerprint(p) for p in sorted(a.output.iterdir()) if p.is_file() and p.name not in ("worker.log","activation.json")}
    save(a.output/"frozen_manifest.json",manifest)
    print("frozen proposal: four heldout topologies, twelve noisy states, forty-eight stress variants; no learning",flush=True)

def verify_frozen(a,cfg):
    assert json.loads((a.frozen/"resolved_config.json").read_text())==cfg
    assert json.loads((a.frozen/"source_manifest.json").read_text())==file_manifest()
    for name,h in json.loads((a.frozen/"frozen_manifest.json").read_text()).items():assert fingerprint(a.frozen/name)==h,name
    return json.loads((a.frozen/"holdout_cases.json").read_text())

def probabilities(model,case,state,t,plan,check):
    tick=time.monotonic();kernel=MarginalKernel(case["prior"],c.schedule(64))
    p0=np.repeat(case["prior"],case["space"].counts).copy();coverage=np.zeros(len(state),int)
    cb=StepPredictor(model,case,state,kernel,t,check_time=check,profile=True)
    try:
        for b in range(4):
            ids,f,q=plan.block(b);p0[ids]=cb(state,ids,f,q);coverage[ids]+=1
        assert np.array_equal(coverage,np.repeat(case["active"].astype(int),case["space"].counts))
        fam=canonical(case)[1];q=kernel.reverse(p0,state,fam,t-1,t)
        if t==1:np.testing.assert_allclose(q,p0,atol=1e-12,rtol=0)
        return p0,q,dict(seconds=time.monotonic()-tick,timings=cb.times,blocks=cb.blocks)
    finally:cb.close()

def probe(a,cfg,records,targets,check):
    case=next(r["case"] for r in records if r["condition"]=="extreme")
    models,initial=pair_models(case,cfg,cfg["train_seeds"][0],a.source);costs={};coverage=[]
    for arm,model in models.items():
        optimizer=opt(model,cfg);model.set_target(targets["extreme"]);updates=[]
        for j in range(5):
            check();normal,robust,mask,receipt=paired_inputs(case,cfg,2026102199,j)
            inp=normal if arm=="normal2" else robust
            value=double_update(model,optimizer,case,inp,64,mask,check)
            updates.append(value);coverage.append(dict(arm=arm,probe_update=j,**receipt))
        dense=np.repeat(case["active"].astype(np.uint8),case["space"].counts);steps=[];model.eval()
        plan=c.plan_for(case,[2026102199,32,61],4)
        for j in range(3):
            check()
            with torch.no_grad():_,_,timing=probabilities(model,case,dense,32,plan,check)
            steps.append(timing)
        values=fixed_validation(records,cfg)
        validation=validate(model,values,cfg,targets,check)
        costs[arm]=dict(updates=updates,actual_dimensions=c.dimensions(model),first_update_seconds=updates[0]["seconds"],
            stable_update_median=float(np.median([v["seconds"] for v in updates[2:]])),
            stable_update_p90=float(np.quantile([v["seconds"] for v in updates[2:]],.9)),
            full_sweep_timings=steps,stable_sweep_p90=float(np.quantile([v["seconds"] for v in steps[1:]],.9)),
            validation_seconds=validation["seconds"])
        print("engineering probe",arm,"stable update",costs[arm]["stable_update_median"],flush=True)
    # Two arms each have3000 optimizer updates,48 T64 chains and12 validations.
    train=3000*sum(v["stable_update_p90"] for v in costs.values())
    sample=48*64*sum(v["stable_sweep_p90"] for v in costs.values())
    val=12*sum(v["validation_seconds"] for v in costs.values())
    # Heldout:12 states×5 variants×3 seeds/arm=180 probability sweeps/arm.
    pressure=180*sum(v["stable_sweep_p90"] for v in costs.values())
    forecast=cfg["forecast_safety_factor"]*(train+sample+val+pressure+60)+cfg["forecast_fixed_overhead_seconds"]
    report=dict(costs=costs,forecast=dict(training_seconds=train,sampling_seconds=sample,validation_seconds=val,
        pressure_seconds=pressure,audit_allowance_seconds=60,total_seconds=forecast),initialization=initial,
        note="disposable engineering updates only; no generated state, warm caches; full-sweep estimate excludes uniform sampling and uses safety factor",
        budget_pass=forecast<cfg["learning_budget_seconds"]-cfg["exit_reserve_seconds"])
    save(a.output/"probe.json",report);save(a.output/"probe_edits.json",coverage)
    return report

def sample_final(model,case,cfg,seed,arm,targets,references,sd,out,check):
    rows=[];model.eval()
    for sample_seed in cfg["sampling_seeds"]:
        for condition in ("balanced","extreme"):
            check();model.set_target(targets[condition]);c.seed_all(sample_seed);rng=np.random.default_rng(sample_seed)
            state=(rng.random(case["space"].total)<np.repeat(case["prior"],case["space"].counts)).astype(np.uint8)
            initial=sha(state.tolist());steps=[];tick=time.monotonic();kernel=MarginalKernel(case["prior"],c.schedule(64))
            for t in range(64,0,-1):
                check();uh,after=uniform_contract(rng,len(state));plan=c.plan_for(case,[sample_seed,t,61],4);h=hashlib.sha256()
                for b in range(4):
                    for ar in plan.block(b):h.update(np.ascontiguousarray(ar).tobytes())
                with torch.no_grad():state,info=reverse_step(model,case,state,kernel,t,4,sample_seed,rng,"cpu",check_time=check)
                assert sha(rng.bit_generator.state)==after
                steps.append(dict(t=t,uniform_sha256=uh,query_sha256=h.hexdigest(),rng_after=after,state_sha256=sha(state.tolist()),edge_count=int(state.sum()),exact_coverage=info["exact_coverage"]))
            edges,fam=c.state_edges(case,state);stem=f"{arm}_{condition}_{sample_seed}"
            path=sd/(stem+".npz");np.savez_compressed(path,types=case["space"].types,edges=edges.T,relations=fam+1,schema=np.asarray(case["space"].pairs))
            save(sd/(stem+"_steps.json"),steps);met=graph(case,edges,NAMES,check);cmp=compare(met,references[condition])
            G=met["typed_degree"]["Genre"]["by_relation"][8];high=int(np.asarray(G["histogram"])[12:].sum())
            row=dict(seed=seed,arm=arm,condition=condition,sampling_seed=sample_seed,initial_state_sha256=initial,
                graph_file=str(path.relative_to(out)),graph_file_sha256=fingerprint(path),metrics=met,comparison=cmp,
                G_W1=cmp["typed_relation_degree_W1"]["Genre"][8],G_joint_TV=cmp["joint_relation_degree_TV"]["Genre"],
                G_max=G["max"],G_high_ge12=high,G_multiple_hubs=high>1,G_low_le1=G["low_le1"],G_zero=G["zero"],seconds=time.monotonic()-tick)
            rows.append(row);append(out/"generation.jsonl",row)
            print("completed frozen-panel sample",seed,arm,condition,sample_seed,flush=True)
    return rows

def pressure_final(models,cfg,targets,frozen,out,seed,check):
    rows=[]
    for rec in json.loads((frozen/"holdout_cases.json").read_text()):
        check();i,t=rec["i"],rec["t"];z=np.load(frozen/rec["file"])
        case=c.make_case(z["types"],z["clean_edges"],tuple(map(tuple,z["schema"])))
        base=dict(edges=z["edges"],edge_families=z["edge_families"])
        plan=c.plan_for(case,[cfg["holdout_query_seed"],i*64+t,21],4)
        clean=np.zeros(case["space"].total,bool);clean[case["clean_ids"]]=True
        ordinary_predictions={}
        for variant in cfg["holdout_variants"]:
            current=base;edit=np.zeros(len(clean),bool)
            if variant!="ordinary":
                zz=np.load(frozen/f"holdout_{i}_t{t}_{variant}.npz")
                current=dict(edges=zz["edges"],edge_families=zz["edge_families"]);edit=zz["edit"]
            state=input_state(case,current)
            for arm,model in models.items():
                check();model.eval();model.set_target(targets[rec["condition"]])
                with torch.no_grad():p,q,detail=probabilities(model,case,state,t,plan,check)
                if variant=="ordinary":ordinary_predictions[arm]=(p.copy(),q.copy())
                name=f"{seed}_{arm}_{i}_t{t}_{variant}.npz";np.savez_compressed(out/name,p0=p,q=q)
                cats={}
                for key,mask in dict(edited=edit,remaining_true=clean&(state==1)&~edit,untouched_nonedge=~clean&~edit).items():
                    op,oq=ordinary_predictions[arm]
                    cats[key]=dict(n=int(mask.sum()),p0=float(p[mask].mean()) if mask.any() else None,q=float(q[mask].mean()) if mask.any() else None,
                        ordinary_p0_same_positions=float(op[mask].mean()) if mask.any() else None,
                        ordinary_q_same_positions=float(oq[mask].mean()) if mask.any() else None)
                rows.append(dict(seed=seed,arm=arm,condition=rec["condition"],id=rec["id"],t=t,variant=variant,
                    categories=cats,expected_edges_p0=float(p.sum()),expected_edges_q=float(q.sum()),
                    births=float(((1-state)*q).sum()),deaths=float((state*(1-q)).sum()),probability_file=name,
                    sha256=fingerprint(out/name),seconds=detail["seconds"]))
    save(out/f"pressure_seed{seed}.json",rows)

def learning(a,cfg,records,targets,check):
    gate=probe(a,cfg,records,targets,check)
    if gate["forecast"]["total_seconds"]>=a.deadline-time.time()-30:raise RuntimeError("complete frozen panel does not fit remaining budget")
    values=fixed_validation(records,cfg);train=[r for r in records if r["split"]=="train"];case=train[0]["case"]
    refs={cond:graph(next(r["case"] for r in records if r["condition"]==cond),next(r["case"] for r in records if r["condition"]==cond)["clean"],NAMES,check) for cond in targets}
    save(a.output/"reference_metrics.json",refs);hashes={};generated=[];total=0
    for seed in cfg["train_seeds"]:
        models,init=pair_models(case,cfg,seed,a.source);optimizers={arm:opt(m,cfg) for arm,m in models.items()}
        assert all(not o.state for o in optimizers.values())
        sd=a.output/f"seed{seed}";sd.mkdir();save(sd/"initialization.json",init)
        for arm,m in models.items():
            path=sd/f"{arm}_initial.pt";hashes[str(path.relative_to(a.output))]=atomic_weights(path,m.state_dict())
        save(a.output/"weight_hashes.json",hashes);rng=np.random.default_rng(seed);completed={arm:0 for arm in models}
        try:
            for u in range(cfg["updates"]+1):
                if u in cfg["validation_updates"]:
                    initial_rows=[]
                    for arm,m in models.items():
                        check();v=validate(m,values,cfg,targets,check);save(sd/f"{arm}_validation_{u:04d}.json",v)
                        if u==0:initial_rows.append(v["rows"])
                    if u==0:assert initial_rows[0]==initial_rows[1],"paired initial validation differs"
                if u==cfg["updates"]:break
                rec=train[(u%2)*4+int(rng.integers(4))];cc=augment(rec["case"],rng)
                normal,robust,mask,corruption=paired_inputs(cc,cfg,seed,u)
                digest=[input_digest(cc,x) for x in normal];changed=input_digest(cc,robust[1])
                for arm,m in models.items():
                    check();m.set_target(targets[rec["condition"]]);inp=normal if arm=="normal2" else robust
                    result=double_update(m,optimizers[arm],cc,inp,64,mask,check)
                    a.successful_updates+=1
                    assert [input_digest(cc,x) for x in normal]==digest and input_digest(cc,robust[1])==changed
                    assert int(optimizers[arm].state[m.backbone.lin_out_E.weight]["step"])==u+1
                    completed[arm]=u+1;total+=1
                    append(a.output/"updates.jsonl",dict(seed=seed,arm=arm,update=u+1,graph_id=rec["id"],condition=rec["condition"],
                        ordinary_input_sha256=digest,perturbed_input_sha256=changed,corruption=corruption,**result))
                if (u+1)%100==0:emit(a.output/"progress.jsonl",dict(seed=seed,paired_update=u+1,total_updates=total,remaining_seconds=a.deadline-time.time()))
            for arm,m in models.items():
                path=sd/f"{arm}_final_1000.pt";hashes[str(path.relative_to(a.output))]=atomic_weights(path,m.state_dict())
            save(a.output/"weight_hashes.json",hashes)
            rr,aa=fixed_response(models,values,targets,cfg,check);save(sd/"condition_response.json",rr);np.savez_compressed(sd/"condition_response.npz",**aa)
            pressure_final(models,cfg,targets,a.frozen,a.output,seed,check)
            for arm,m in models.items():generated.extend(sample_final(m,case,cfg,seed,arm,targets,refs,sd,a.output,check))
        except BaseException:
            atomic_weights(a.output/"interrupted_state.pt",dict(seed=seed,completed=completed,models={arm:m.state_dict() for arm,m in models.items()},
                optimizers={arm:o.state_dict() for arm,o in optimizers.items()},rng=c.rng_save(rng,"cpu")))
            raise
        models={};optimizers={};gc.collect()
    assert total==6000 and len(generated)==96
    save(a.output/"generation.json",generated)
    from evaluate import audit
    audit(a.output,cfg,check)
    return total,len(generated)

def main():
    p=argparse.ArgumentParser();p.add_argument("mode",choices=("freeze","probe","panel"))
    for key in ("source","panel","output"):p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--frozen",type=Path);p.add_argument("--deadline",type=float,required=True);a=p.parse_args()
    started=time.monotonic();cpu=time.process_time();setup();cfg=config();status="failed";reason=None;updates=0;chains=0;a.successful_updates=0
    def check():
        if time.time()>a.deadline-(30 if a.mode=="panel" else 5):raise TimeoutError("exit reserve")
        if (a.output/"STOP").exists():raise InterruptedError("STOP")
    def interrupted(sig,frame):raise InterruptedError(f"worker signal {sig}")
    for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(sig,interrupted)
    try:
        check();bindings=check_source(a.source,a.panel,cfg);records,targets=load_panel(a.panel)
        save(a.output/"input_bindings.json",bindings);save(a.output/"environment.json",c.environment("cpu"))
        if a.mode=="freeze":freeze(a,cfg,records,targets,check)
        else:
            verify_frozen(a,cfg)
            save(a.output/"resolved_config.json",cfg);save(a.output/"source_manifest.json",file_manifest())
            save(a.output/"frozen_binding.json",dict(path=str(a.frozen),manifest_sha256=fingerprint(a.frozen/"frozen_manifest.json")))
            if a.mode=="probe":probe(a,cfg,records,targets,check)
            else:updates,chains=learning(a,cfg,records,targets,check)
        check();assert all(fingerprint(path)==h for path,h in bindings.items())
        if a.mode!="freeze":verify_frozen(a,cfg)
        status="completed";return 0
    except BaseException as exc:
        reason=repr(exc);status="stopped" if isinstance(exc,(InterruptedError,TimeoutError)) else "failed";raise
    finally:
        chain_file=a.output/"generation.jsonl"
        saved_chains=len(chain_file.read_text().splitlines()) if chain_file.exists() else 0
        save(a.output/"worker_receipt.json",dict(status=status,reason=reason,mode=a.mode,scientific_updates=a.successful_updates,complete_chains=saved_chains,
            engineering_updates=10 if a.mode in ("probe","panel") and (a.output/"probe.json").exists() else 0,
            wall_seconds=time.monotonic()-started,CPU_seconds=time.process_time()-cpu,
            peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,GPU=0,
            learning_panel_activated=a.mode=="panel",partial_updates_log="updates.jsonl" if (a.output/"updates.jsonl").exists() else None))
if __name__=="__main__":sys.exit(main())
