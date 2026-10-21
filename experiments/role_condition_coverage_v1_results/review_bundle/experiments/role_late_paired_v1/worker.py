"""Frozen two-arm experiment. Never resume/reseed an incomplete panel."""
import json
import sys
from pathlib import Path
import time
import resource
import hashlib
import gc
import numpy as np
import torch
from pairing import BASE,ROOT,c,config,construct,input_digest,uniform_contract,atomic_weights,source_manifest,sha
from hsgd.runtime import reverse_step,predict
from hsgd.kernel import MarginalKernel
from role_condition.experiment import save,emit,opt,fixed_validation,validate,set_condition
from role_condition.panel import make_panel,save_panel,augment,NAMES
from role_condition.metrics import graph,compare
from hsgd.computational import FrozenTypedContext


def append(path,row):
    with path.open("a") as f:f.write(json.dumps(row,allow_nan=False)+"\n")


def response(models,values,targets,cfg,check):
    rows=[];arrays={}
    for arm,m in models.items():
        m.eval()
        for identifier,cond,case,inp,index in values[:3]:
            plan=c.plan_for(case,[cfg["validation_seed"],index,21],4);ps=[];ls=[];mg=None
            for target in ["balanced","extreme","balanced"]:
                check();m.set_target(targets[target]);ll=[];pp=[];ff=[];ii=[]
                ctx=FrozenTypedContext(m,case["space"].types,inp["edges"],inp["edge_families"],"response")
                try:
                    with torch.no_grad():
                        for block in range(4):
                            ids,f,q=plan.block(block);_,l,p=ctx.predict(q,f,inp["t"]/64,"response",details=True)
                            ii.append(ids);ll.append(l.numpy());pp.append(p.numpy());ff.append(f)
                    order=np.argsort(np.concatenate(ii));ls.append(np.concatenate(ll)[order]);ps.append(np.concatenate(pp)[order])
                    mg=np.concatenate(ff)[order]==case["space"].pairs.index((2,3))
                finally:ctx.close()
            delta=ps[1]-ps[0];gap=(ls[1][:,1]-ls[1][:,0])-(ls[0][:,1]-ls[0][:,0])
            row=dict(arm=arm,t=inp["t"],input_id=identifier,
                p0_max_abs=float(abs(delta).max()),MG_p0_max_abs=float(abs(delta[mg]).max()),
                MG_p0_mean_abs=float(abs(delta[mg]).mean()),MG_logit_gap_max_abs=float(abs(gap[mg]).max()),
                repeat_max_abs=float(abs(ps[2]-ps[0]).max()))
            rows.append(row)
            for j,p in enumerate(ps):arrays[f"{arm}_t{inp['t']}_target{j}_p0"]=p
            for j,l in enumerate(ls):arrays[f"{arm}_t{inp['t']}_target{j}_logits"]=l
    return rows,arrays


def main(mode,out,deadline):
    out=Path(out);cfg=config()
    torch.set_num_threads(1);torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if mode=="freeze":
        import subprocess
        save(out/"source_manifest.json",source_manifest());save(out/"resolved_config.json",cfg)
        save(out/"environment.json",c.environment("cpu"))
        save(out/"code_revision.json",dict(commit=subprocess.check_output(
            ["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip(),
            base_commit=cfg["base_commit"]))
        return 0
    def check():
        if (out/"STOP").exists():raise InterruptedError("STOP")
        if time.monotonic()>deadline-30:raise TimeoutError("single cumulative budget reserve")
    assert json.loads((out/"source_manifest.json").read_text())==source_manifest()
    assert json.loads((out/"resolved_config.json").read_text())==cfg
    started=time.monotonic();cpu=time.process_time();status="failed";reason=None;stage="preparation"
    models={};optimizers={};completed={};seed=None
    try:
        records,targets=make_panel(cfg["panel_seed"]);save_panel(records,targets,out/"panel.json")
        case=records[0]["case"];values=fixed_validation(records,cfg);train=[r for r in records if r["split"]=="train"]
        references={}
        for cond in targets:
            ref=next(r["case"] for r in records if r["condition"]==cond)
            references[cond]=graph(ref,ref["clean"],NAMES,check)
        save(out/"reference_metrics.json",references)
        # Cost gate uses disposable weights, never the scientific initial state.
        stage="cost_gate";probe_models,_=construct(case,cfg,2026102099);costs={}
        probe_case=next(r["case"] for r in records if r["condition"]=="extreme")
        inps=[c.train_input(probe_case,64,4,2026102099,i,np.random.default_rng([2026102099,i]),force_t=32) for i in range(5)]
        for arm,m in probe_models.items():
            m.set_target(targets["extreme"]);optimizer=opt(m,cfg);dt=[]
            for inp in inps:
                check();tick=time.perf_counter();c.update(m,optimizer,probe_case,inp,64,"cpu");dt.append(time.perf_counter()-tick)
            m.eval();dense=np.repeat(case["active"].astype(np.uint8),case["space"].counts);st=[]
            for i in range(3):
                check();tick=time.perf_counter()
                with torch.no_grad():reverse_step(m,case,dense,MarginalKernel(case["prior"],c.schedule(64)),32,4,12,np.random.default_rng(12),"cpu",check_time=check)
                st.append(time.perf_counter()-tick)
            v=validate(m,values,cfg,targets,check)
            costs[arm]=dict(updates_seconds=dt,stable_p90=float(np.quantile(dt[2:],.9)),dense_steps=st,
                            step_p90=float(np.quantile(st[1:],.9)),validation_seconds=v["seconds"])
            del optimizer
        forecast=1.15*(3000*sum(x["stable_p90"] for x in costs.values())+
                48*64*sum(x["step_p90"] for x in costs.values())+
                12*sum(x["validation_seconds"] for x in costs.values()))+90
        remaining=deadline-time.monotonic()
        save(out/"cost_gate.json",dict(costs=costs,forecast_seconds=forecast,remaining_seconds=remaining,
             passed=forecast<remaining-30,scope="full model dense-state step cost; no quality selection"))
        del probe_models,m;gc.collect()
        if forecast>=remaining-30:raise RuntimeError("cost gate rejects full frozen scope; no scientific learning")
        stage="learning_sampling";generations=[];hashes={};all_completed=0
        for seed in cfg["train_seeds"]:
            check();models,init=construct(case,cfg,seed);optimizers={a:opt(m,cfg) for a,m in models.items()}
            assert all(not o.state for o in optimizers.values())
            assert len({id(o) for o in optimizers.values()})==2
            completed={a:0 for a in models};sd=out/f"seed{seed}";sd.mkdir()
            for arm,m in models.items():
                path=sd/f"{arm}_initial.pt";hashes[str(path.relative_to(out))]=atomic_weights(path,m.state_dict())
            save(sd/"initialization.json",init);save(out/"weight_hashes.json",hashes)
            rng=np.random.default_rng(seed)
            for u in range(1001):
                if u in [0,250,500,1000]:
                    for arm,m in models.items():
                        check();v=validate(m,values,cfg,targets,check)
                        save(sd/f"{arm}_validation_{u:04d}.json",dict(seed=seed,arm=arm,update=u,**v))
                        emit(out/"progress.jsonl",dict(phase="validation",seed=seed,arm=arm,update=u,ce=v["mean_ce"]))
                if u==1000:break
                check();rec=train[(u%2)*4+int(rng.integers(4))];current=augment(rec["case"],rng)
                inp=c.train_input(current,64,4,seed,u,rng);digest=input_digest(current,inp)
                begin_rng=torch.get_rng_state();after_rng=None
                for arm,m in models.items():
                    check();m.set_target(targets[rec["condition"]]);torch.set_rng_state(begin_rng)
                    tick=time.perf_counter();result=c.update(m,optimizers[arm],current,inp,64,"cpu");dt=time.perf_counter()-tick
                    assert input_digest(current,inp)==digest,"training input mutated"
                    key=m.backbone.lin_out_E.weight
                    assert int(optimizers[arm].state[key]["step"])==u+1
                    now=torch.get_rng_state()
                    if after_rng is None:after_rng=now
                    else:assert torch.equal(now,after_rng),"training RNG consumption differs"
                    completed[arm]=u+1;all_completed+=1
                    append(out/"updates.jsonl",dict(seed=seed,arm=arm,update=u+1,input_sha256=digest,
                        torch_rng_before=hashlib.sha256(begin_rng.numpy().tobytes()).hexdigest(),
                        torch_rng_after=hashlib.sha256(now.numpy().tobytes()).hexdigest(),
                        graph_id=rec["id"],condition=rec["condition"],t=inp["t"],query_block=inp["round_id"],
                        query_count=len(inp["ids"]),loss=result["loss"],gradient_norm=result["gradient_norm"],
                        seconds=dt,optimizer_step=int(optimizers[arm].state[key]["step"])))
                if (u+1)%100==0:emit(out/"progress.jsonl",dict(phase="training",seed=seed,paired_update=u+1,
                     effective_total_updates=all_completed,remaining_seconds=deadline-time.monotonic()))
            for arm,m in models.items():
                path=sd/f"{arm}_final_1000.pt";hashes[str(path.relative_to(out))]=atomic_weights(path,m.state_dict())
            save(out/"weight_hashes.json",hashes)
            rr,aa=response(models,values,targets,cfg,check);save(sd/"condition_response.json",rr)
            np.savez_compressed(sd/"condition_response.npz",**aa)
            for arm,m in models.items():
                m.eval()
                for sample_seed in cfg["sampling_seeds"]:
                    for cond in ["balanced","extreme"]:
                        check();m.set_target(targets[cond]);c.seed_all(sample_seed);sampling=np.random.default_rng(sample_seed)
                        state=(sampling.random(case["space"].total)<np.repeat(case["prior"],case["space"].counts)).astype(np.uint8)
                        initial_hash=sha(state.tolist());step_records=[];tick=time.perf_counter()
                        kernel=MarginalKernel(case["prior"],c.schedule(64))
                        for t in range(64,0,-1):
                            check();uniform_sha,expected_rng=uniform_contract(sampling,case["space"].total)
                            plan=c.plan_for(case,[sample_seed,t,61],4);qh=hashlib.sha256()
                            for j in range(4):
                                for arr in plan.block(j):qh.update(np.ascontiguousarray(arr).tobytes())
                            with torch.no_grad():state,info=reverse_step(m,case,state,kernel,t,4,sample_seed,sampling,"cpu",check_time=check)
                            assert sha(sampling.bit_generator.state)==expected_rng
                            step_records.append(dict(t=t,uniform_sha256=uniform_sha,query_sha256=qh.hexdigest(),
                                rng_after=expected_rng,state_sha256=sha(state.tolist()),edge_count=int(state.sum()),
                                exact_coverage=info["exact_coverage"],blocks=info["blocks"],seconds=info["step_seconds"]))
                        edges,fam=c.state_edges(case,state);stem=f"{arm}_{cond}_{sample_seed}"
                        np.savez_compressed(sd/(stem+".npz"),types=case["space"].types,edges=edges.T,
                            relations=fam+1,schema=np.asarray(case["space"].pairs))
                        save(sd/(stem+"_steps.json"),step_records)
                        metrics=graph(case,edges,NAMES,check);comparison=compare(metrics,references[cond])
                        mg=case["space"].pairs.index((2,3));g=metrics["typed_degree"]["Genre"]["by_relation"][mg]
                        hist=np.array(g["histogram"]);high=int(hist[12:].sum())
                        row=dict(seed=seed,arm=arm,condition=cond,sampling_seed=sample_seed,
                            graph_file=str((sd/(stem+".npz")).relative_to(out)),graph_file_sha256=c.sha(sd/(stem+".npz")),
                            initial_state_sha256=initial_hash,metrics=metrics,comparison=comparison,
                            G_W1=comparison["typed_relation_degree_W1"]["Genre"][mg],
                            G_joint_TV=comparison["joint_relation_degree_TV"]["Genre"],G_max=g["max"],
                            G_high_ge12=high,G_multiple_hubs=high>1,G_zero=g["zero"],G_low_le1=g["low_le1"],
                            seconds=time.perf_counter()-tick)
                        generations.append(row);append(out/"generation.jsonl",row)
                        emit(out/"progress.jsonl",dict(phase="sample",seed=seed,arm=arm,condition=cond,
                             sampling_seed=sample_seed,edges=metrics["edges"],G_max=g["max"],
                             complete_chains=len(generations),remaining_seconds=deadline-time.monotonic()))
            del m;models={};optimizers={};gc.collect()
        assert all_completed==6000 and len(generations)==96
        save(out/"generation.json",generations)
        assert source_manifest()==json.loads((out/"source_manifest.json").read_text())
        status="completed";return 0
    except BaseException as exc:
        reason=repr(exc)
        if isinstance(exc,(TimeoutError,InterruptedError)):status="stopped"
        if models:
            # Partial paired update is retained with individual counters, not
            # promoted to a completed pair. No retry/resume path in this panel.
            atomic_weights(out/"interrupted_state.pt",dict(seed=seed,completed=completed,
                models={a:m.state_dict() for a,m in models.items()},
                optimizers={a:o.state_dict() for a,o in optimizers.items()},torch_rng=torch.get_rng_state()))
        raise
    finally:
        save(out/"worker_receipt.json",dict(status=status,reason=reason,stage=stage,
            wall_seconds=time.monotonic()-started,CPU_seconds=time.process_time()-cpu,
            peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            device="CPU",no_budget_reset=True,partial_seed=seed,partial_completed=completed))
if __name__=="__main__":sys.exit(main(sys.argv[1],sys.argv[2],float(sys.argv[3])))
