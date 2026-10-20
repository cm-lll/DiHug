"""Fixed-weight probability diagnostics. No optimizer or graph sampler."""
import os
os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
import argparse
import hashlib
import json
from pathlib import Path
import resource
import signal
import sys
import time
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP/"role_specialization_v1"))
from panel import CapacityDenoiser, load_panel, c, save, fingerprint, fixed_validation, validate
from hsgd.runtime import StepPredictor
from hsgd.kernel import MarginalKernel
from hsgd.noising import global_noisy_edges

BASELINE = "b8e97cc42e6e835e0ea228a6666e3a22a200e035"
SEEDS = [2026102011, 2026102012, 2026102013]
TIMES = [1, 4, 8]

def slots(case):
    sp = case["space"]
    edges = np.concatenate([sp.decode(f, np.arange(n)) for f,n in enumerate(sp.counts)], axis=1)
    fam = np.repeat(np.arange(case["n_families"]), sp.counts)
    return edges, fam

def degrees(values, edges, fam, n, R):
    out = np.zeros((n, R))
    for side in (0,1):
        np.add.at(out, (edges[side],fam), values)
    return out

def edit_variants(case, condition, clean, ordinary, selected, rng):
    edges,fam = slots(case); mg = case["space"].pairs.index((2,3))
    movable = (case["prior"][fam]>0)&(case["prior"][fam]<1)
    clean_deg = degrees(clean,edges,fam,len(case["space"].types),case["n_families"])
    G = np.flatnonzero(case["space"].types==3)
    allowed = G[clean_deg[G,mg]==1] if condition=="extreme" else G
    incident = (edges==selected).any(0)
    candidates = (fam==mg)&movable
    candidates &= (~clean.astype(bool)&~ordinary.astype(bool)) if condition=="extreme" else (clean.astype(bool)&ordinary.astype(bool))
    local = rng.permutation(np.flatnonzero(candidates&incident))
    others = rng.permutation(allowed[allowed!=selected])
    queues = [list(rng.permutation(np.flatnonzero(candidates&(edges==g).any(0)))) for g in others]
    spread = []
    while any(queues):
        for queue in queues:
            if queue: spread.append(queue.pop())
    current = int(ordinary[incident&(fam==mg)].sum())
    demands = [("add1",1),("to5",5-current),("to6",6-current)] if condition=="extreme" else [("delete1",1),("delete2",2),("delete3",3)]
    rows = []
    for strength,k in demands:
        for location,pool in [("concentrated",local),("distributed",np.asarray(spread,dtype=int))]:
            reason = None
            if k<=0: reason = "requested target degree not above ordinary degree"
            elif len(local)<k: reason = "paired concentrated edit infeasible"
            elif len(pool)<k: reason = "same-count location control infeasible"
            row = dict(strength=strength,location=location,requested_edits=k,selected_node=int(selected),
                       selected_ordinary_degree=current,status="infeasible" if reason else "ready",reason=reason)
            if reason:
                rows.append((row,None,None)); continue
            mask=np.zeros(len(clean),dtype=bool);mask[pool[:k]]=True
            state=ordinary.copy();state[mask]=1 if condition=="extreme" else 0
            assert np.count_nonzero(state!=ordinary)==k
            assert movable[mask].all()
            rows.append((row,state,mask))
    return rows

def query_digest(plan):
    h=hashlib.sha256()
    for b in range(plan.rounds):
        for a in plan.block(b): h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()

def sweep(model,case,state,kernel,t,plan,check):
    start=time.monotonic(); p0=np.repeat(case["prior"],case["space"].counts).copy()
    covered=np.zeros(len(state),dtype=int)
    before=torch.get_rng_state().clone()
    cb=StepPredictor(model,case,state,kernel,t,check_time=check,profile=True)
    try:
        for b in range(plan.rounds):
            ids,fam,query=plan.block(b)
            p0[ids]=cb(state,ids,fam,query);covered[ids]+=1
        assert np.array_equal(covered,np.repeat(case["active"].astype(int),case["space"].counts))
        fam=slots(case)[1]; q=kernel.reverse(p0,state,fam,t-1,t)
        assert np.isfinite(q).all() and ((q>=0)&(q<=1)).all()
        if t==1: np.testing.assert_allclose(q,p0,atol=1e-12,rtol=0)
        assert torch.equal(before,torch.get_rng_state())
        return p0,q,dict(seconds=time.monotonic()-start,blocks=cb.blocks,coverage=covered.tolist(),timings=cb.times)
    finally: cb.close()

def metrics(case,clean,state,edit,selected,p0,q):
    edges,fam=slots(case);mg=case["space"].pairs.index((2,3))
    y=clean.astype(bool); x=state.astype(bool)
    masks=dict(edited=edit,added_false=edit&~y,deleted_true=edit&y,
               remaining_true=y&x&~edit,untouched_nonedge=~y&~edit)
    masks.update({k+"_MG":v&(fam==mg) for k,v in list(masks.items())})
    categories={}
    for name,mask in masks.items():
        categories[name]=dict(n=int(mask.sum()),p0=float(p0[mask].mean()) if mask.any() else None,
                            q=float(q[mask].mean()) if mask.any() else None)
    birth=(1-state)*q;death=state*(1-q)
    local=(edges==selected).any(0)
    nodes=np.unique(edges[:,edit]) if edit.any() else np.array([selected])
    edited_local=np.isin(edges,nodes).any(0)
    flows={}
    for name,mask in [("global",np.ones(len(x),bool)),("selected_node",local),("edited_endpoints",edited_local),("MG",fam==mg)]:
        flows[name]=dict(births=float(birth[mask].sum()),deaths=float(death[mask].sum()),
                        net=float((birth-death)[mask].sum()),current_edges=int(state[mask].sum()),
                        expected_next_edges=float(q[mask].sum()))
        np.testing.assert_allclose(flows[name]["net"],float((q-state)[mask].sum()),atol=1e-12)
    G=np.flatnonzero(case["space"].types==3)
    target=degrees(clean,edges,fam,len(case["space"].types),case["n_families"])
    role={}
    for name,values in [("p0",p0),("q",q)]:
        mu=degrees(values,edges,fam,len(case["space"].types),case["n_families"])
        # Joint empirical TV over exact vectors; not separate marginal matching.
        from collections import Counter
        a=Counter(map(tuple,mu[G]));b=Counter(map(tuple,target[G]))
        tv=sum(abs(a[k]-b[k]) for k in a.keys()|b.keys())/(2*len(G))
        role[name]=dict(expected_degree=mu.tolist(),G_expected_W1=float(np.mean(abs(np.sort(mu[G,mg])-np.sort(target[G,mg])))),
            G_joint_expected_TV=float(tv),G_expected_max=float(mu[G,mg].max()),
            G_target_multiset=sorted(target[G,mg].tolist()))
    return dict(categories=categories,flows=flows,role=role,
        by_family=dict(expected_p0=np.bincount(fam,weights=p0,minlength=case["n_families"]).tolist(),
        expected_q=np.bincount(fam,weights=q,minlength=case["n_families"]).tolist(),
        births=np.bincount(fam,weights=birth,minlength=case["n_families"]).tolist(),
        deaths=np.bincount(fam,weights=death,minlength=case["n_families"]).tolist()))

def compare_validation(old,new):
    assert len(old["rows"])==len(new["rows"])==12
    maximum=0.
    for a,b in zip(old["rows"],new["rows"]):
        for key in ("id","condition","t","counts_by_family_label"): assert a[key]==b[key]
        for key in ("ce","ce_sum_by_family_label","expected_edges","expected_edge_bias"):
            aa=np.asarray(a[key]);bb=np.asarray(b[key])
            np.testing.assert_allclose(aa,bb,atol=1e-6,rtol=1e-5)
            maximum=max(maximum,float(np.max(abs(aa-bb))))
    return dict(max_absolute_error=maximum,old_mean_ce=old["mean_ce"],new_mean_ce=new["mean_ce"])

def run(args):
    start=time.monotonic();cpu=time.process_time()
    args.output.mkdir(parents=True,exist_ok=False)
    status="failed";reason=None;rows=[];infeasible=[];forward_sweeps=0
    def check():
        if time.monotonic()-start>595: raise TimeoutError("600 second diagnostic budget reserve")
        if (args.output/"STOP").exists(): raise InterruptedError("STOP")
    def interrupted(sig,frame): raise InterruptedError(f"signal {sig}")
    for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP): signal.signal(sig,interrupted)
    torch.set_num_threads(1);torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    sources={}
    def bind(p):
        sources[str(p)]=fingerprint(p);return json.loads(p.read_text())
    try:
        historical=EXP/"role_specialization_v1_results/evidence/learning_run01"
        required=["resolved_config.json","weight_hashes.json","environment.json","source_manifest.json"]
        required += [f"seed{seed}/capacity_C_validation_1000.json" for seed in SEEDS]
        for name in required: assert fingerprint(args.run/name)==fingerprint(historical/name),name
        assert fingerprint(args.panel/"panel.json")=="d69e554a3ec43bea1b039bc25f45bf9f4b49e1485dae4b2504fcbc7ce1df9bf0"
        cfg=bind(args.run/"resolved_config.json"); assert cfg["train_seeds"]==SEEDS
        assert cfg["T"]==64 and cfg["rounds"]==4
        weights=bind(args.run/"weight_hashes.json")
        archived=json.loads((EXP/"role_specialization_v1_results/evidence/learning_run01/weight_hashes.json").read_text())
        assert weights==archived
        env=c.environment("cpu");oldenv=bind(args.run/"environment.json")
        for key in ("python_version","packages","torch_runtime_version","float32_matmul_precision","matmul_allow_tf32","deterministic_algorithms","torch_threads"):
            assert env[key]==oldenv[key],key
        save(args.output/"environment.json",env)
        manifest=bind(args.run/"source_manifest.json")
        for name,h in manifest.items():
            assert fingerprint(EXP/name)==h,name
        bind(args.panel/"panel.json")
        records,targets=load_panel(args.panel);case=records[0]["case"]
        values=fixed_validation(records,cfg)
        panel=[];frozen=[]
        for i,rec in enumerate(r for r in records if r["split"]=="validation"):
            cc=rec["case"];ee,ff=slots(cc);mg=cc["space"].pairs.index((2,3))
            clean=np.zeros(cc["space"].total,dtype=np.uint8);clean[cc["clean_ids"]]=1
            deg=degrees(clean,ee,ff,len(cc["space"].types),cc["n_families"])
            G=np.flatnonzero(cc["space"].types==3);eligible=G[deg[G,mg]==1] if rec["condition"]=="extreme" else G
            selected=int(np.random.default_rng([2026102102,i]).choice(eligible))
            for t in TIMES:
                ne,_,_=global_noisy_edges(cc["space"],cc["clean"],cc["prior"],c.schedule(64)[t],np.random.default_rng([2026102101,i,t]))
                nf,nl=cc["space"].encode(ne);state=np.zeros(len(clean),dtype=np.uint8);state[cc["space"].offsets[nf]+nl]=1
                plan=c.plan_for(cc,[2026102103,i,t],4)
                meta=dict(graph=rec["id"],condition=rec["condition"],t=t,selected_node=selected,query_sha256=query_digest(plan))
                stem=f"{rec['id']}_t{t}"
                variants=edit_variants(cc,rec["condition"],clean,state,selected,np.random.default_rng([2026102104,i,t]))
                arrays=dict(clean=clean,ordinary=state,types=cc["space"].types,endpoints=ee,families=ff)
                for b in range(4):
                    ids,fam,query=plan.block(b)
                    arrays.update({f"ids{b}":ids,f"families{b}":fam,f"query{b}":query})
                for info,xx,mask in variants:
                    panel.append(dict(**meta,**{k:v for k,v in info.items() if k!="selected_node"}))
                    if xx is None: infeasible.append(panel[-1]);continue
                    name=info["strength"]+"_"+info["location"];arrays[name]=xx;arrays[name+"_edit"]=mask
                np.savez_compressed(args.output/(stem+"_input.npz"),**arrays)
                frozen.append((cc,meta,stem,clean,state,selected,plan,variants))
        save(args.output/"panel.json",panel)
        save(args.output/"resolved_config.json",dict(baseline=BASELINE,training=cfg,diagnostic_seeds=[2026102101,2026102102,2026102103,2026102104],
             times=TIMES,max_seconds=600,protocol_sha256=fingerprint(HERE/"DESIGN.md"),probability_only=True))
        save(args.output/"source_hashes.json",{str(p.relative_to(EXP)):fingerprint(p) for p in sorted(HERE.glob("*")) if p.is_file()})
        validation=[]
        for seed in SEEDS:
            check();path=args.run/f"seed{seed}/capacity_C_final_1000.pt"
            assert fingerprint(path)==weights[str(path.relative_to(args.run))]
            sources[str(path)]=fingerprint(path)
            model=CapacityDenoiser(case["n_types"],case["n_families"],family_pairs=case["space"].pairs,prior=case["prior"],T=64,**cfg["model"])
            model.load_state_dict(torch.load(path,map_location="cpu",weights_only=True),strict=True);model.eval()
            old=bind(args.run/f"seed{seed}/capacity_C_validation_1000.json")
            v=validate(model,values,cfg,targets,check);replay=compare_validation(old,v)
            validation.append(dict(seed=seed,**replay,seconds=v["seconds"]))
            save(args.output/f"validation_{seed}.json",v);save(args.output/"validation_replay.json",validation)
            print("validation reproduced",seed,replay,flush=True)
            if seed==SEEDS[0]:
                estimate=2*(2*v["seconds"]+255*v["seconds"]/12)+30
                gate=dict(forecast_seconds=estimate,remaining_seconds=595-(time.monotonic()-start),passed=estimate<595-(time.monotonic()-start))
                save(args.output/"cost_gate.json",gate)
                if not gate["passed"]: raise RuntimeError("finite panel exceeds budget")
            for index,(cc,meta,stem,clean,state,selected,plan,variants) in enumerate(frozen):
                check();model.set_target(targets[meta["condition"]]);kernel=MarginalKernel(cc["prior"],c.schedule(64))
                p,q,info=sweep(model,cc,state,kernel,meta["t"],plan,check);forward_sweeps+=1
                np.savez_compressed(args.output/f"{seed}_{stem}_ordinary.npz",p0=p,q=q)
                if index==0:
                    pp,qq,_=sweep(model,cc,state,kernel,meta["t"],plan,check);forward_sweeps+=1
                    np.testing.assert_allclose(pp,p,atol=1e-7,rtol=0);np.testing.assert_allclose(qq,q,atol=1e-7,rtol=0)
                    save(args.output/f"repeat_{seed}.json",dict(p0_max_abs=float(max(abs(pp-p))),q_max_abs=float(max(abs(qq-q)))))
                rows.append(dict(seed=seed,**meta,variant="ordinary",timing=info,
                    metrics=metrics(cc,clean,state,np.zeros(len(clean),bool),selected,p,q)))
                for variant,xx,mask in variants:
                    if xx is None: continue
                    check();pp,qq,detail=sweep(model,cc,xx,kernel,meta["t"],plan,check);forward_sweeps+=1
                    name=variant["strength"]+"_"+variant["location"]
                    np.savez_compressed(args.output/f"{seed}_{stem}_{name}.npz",p0=pp,q=qq)
                    rows.append(dict(seed=seed,**meta,variant=name,edits=int(mask.sum()),timing=detail,
                        metrics=metrics(cc,clean,xx,mask,selected,pp,qq),
                        paired_ordinary=metrics(cc,clean,state,mask,selected,p,q)))
                print("probability panel",seed,stem,"sweeps",forward_sweeps,flush=True)
                save(args.output/"records.json",rows)
            assert fingerprint(path)==sources[str(path)]
        assert all(fingerprint(p)==h for p,h in sources.items())
        save(args.output/"sources.json",sources);status="completed"
    except BaseException as exc:
        reason=repr(exc);raise
    finally:
        save(args.output/"receipt.json",dict(status=status,reason=reason,records=len(rows),infeasible=infeasible,
            probability_sweeps=forward_sweeps,training_updates=0,generated_graphs=0,GPU=0,
            wall_seconds=time.monotonic()-start,CPU_seconds=time.process_time()-cpu,
            peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024))

def main():
    parser=argparse.ArgumentParser()
    for key in ("run","panel","output"):parser.add_argument("--"+key,type=Path,required=True)
    run(parser.parse_args())

if __name__=="__main__":main()
