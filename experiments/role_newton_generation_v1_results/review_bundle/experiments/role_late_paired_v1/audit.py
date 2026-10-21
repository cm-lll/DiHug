"""Independent raw-graph / paired-input audit, summaries and descriptive curves."""
import collections
import itertools
import json
from pathlib import Path
import resource
import signal
import sys
import time
import numpy as np
import networkx as nx
import torch
from pairing import c,config,source_manifest
from role_condition.experiment import save


def lines(p):return [json.loads(x) for x in p.read_text().splitlines()]


def main(out,deadline):
    out=Path(out);started=time.perf_counter();cpu=time.process_time()
    def check():
        if (out/"STOP").exists() or time.monotonic()>deadline-10:raise TimeoutError("audit cumulative deadline/STOP")
    check();cfg=config();assert source_manifest()==json.loads((out/"source_manifest.json").read_text())
    updates=lines(out/"updates.jsonl");rows=json.loads((out/"generation.json").read_text())
    assert len(updates)==6000 and len(rows)==96
    weights=json.loads((out/"weight_hashes.json").read_text());assert len(weights)==12
    for name,h in weights.items():check();assert c.sha(out/name)==h
    for seed in cfg["train_seeds"]:
        sd=out/f"seed{seed}";init=json.loads((sd/"initialization.json").read_text())
        a=torch.load(sd/"original_C_initial.pt",map_location="cpu",weights_only=True)
        b=torch.load(sd/"late_C_initial.pt",map_location="cpu",weights_only=True)
        assert all(torch.equal(a[k],b[k]) for k in init["shared_tensors"]);del a,b
        for u in range(1,1001):
            paired=[x for x in updates if x["seed"]==seed and x["update"]==u]
            assert len(paired)==2 and {x["arm"] for x in paired}==set(cfg["arms"])
            for k in ["input_sha256","torch_rng_before","torch_rng_after","t","query_block","query_count"]:
                assert paired[0][k]==paired[1][k],k
            assert all(x["optimizer_step"]==u for x in paired)
    refs=json.loads((out/"reference_metrics.json").read_text());graphs={};checks=[]
    for row in rows:
        check();path=out/row["graph_file"];assert c.sha(path)==row["graph_file_sha256"]
        with np.load(path,allow_pickle=False) as z:
            types=z["types"];edges=z["edges"];fam=z["relations"]-1;pairs=z["schema"]
        assert len(types)==34 and len(edges)==len({tuple(e) for e in edges})
        assert (edges[:,0]<edges[:,1]).all()
        assert np.array_equal(pairs[fam],np.sort(types[edges],axis=1))
        g=nx.Graph();g.add_nodes_from((i,{"type":int(t)}) for i,t in enumerate(types))
        g.add_edges_from((int(u),int(v),{"relation":int(r)}) for (u,v),r in zip(edges,fam))
        m=row["metrics"];assert g.number_of_edges()==m["edges"]
        assert sum(nx.triangles(g).values())//3==m["triangles"]
        assert sorted(map(len,nx.connected_components(g)),reverse=True)==m["component_sizes"]
        degree=np.zeros((34,len(pairs)),dtype=int)
        for (u,v),r in zip(edges,fam):degree[u,r]+=1;degree[v,r]+=1
        assert int(np.max(degree.sum(1)))==m["maximum_degree"]
        for typ,name in enumerate(["Director","Actor","Movie","Genre"]):
            joint={",".join(map(str,k)):v for k,v in collections.Counter(map(tuple,degree[types==typ])).items()}
            assert joint==m["joint_relation_degree"][name]
        genre=degree[types==3,8];target=refs[row["condition"]]["typed_degree"]["Genre"]["by_relation"][8]["histogram"]
        target_degrees=np.repeat(np.arange(len(target)),target)
        w1=float(np.abs(np.sort(genre)-np.sort(target_degrees)).mean())
        assert abs(w1-row["G_W1"])<1e-12
        assert int((genre>=12).sum())==row["G_high_ge12"]
        assert int((genre<=1).sum())==row["G_low_le1"]
        closure_num=closure_den=0
        for movie in np.flatnonzero(types==2):
            ds=[v for v in g[movie] if types[v]==0];acts=[v for v in g[movie] if types[v]==1]
            for d in ds:
                for actor in acts:closure_den+=1;closure_num+=g.has_edge(d,actor)
        assert (closure_num,closure_den)==(m["imdb_closure"]["path_numerator"],m["imdb_closure"]["path_denominator"])
        steps=json.loads(path.with_name(path.stem+"_steps.json").read_text())
        assert [s["t"] for s in steps]==list(range(64,0,-1))
        assert all(s["blocks"]==4 and s["exact_coverage"] for s in steps)
        key=(row["seed"],row["arm"],row["condition"],row["sampling_seed"])
        graphs[key]=g;checks.append(dict(graph_file=row["graph_file"],passed=True))
    for seed in cfg["train_seeds"]:
        for sample in cfg["sampling_seeds"]:
            records=[]
            for arm,cond in itertools.product(cfg["arms"],["balanced","extreme"]):
                path=out/f"seed{seed}"/f"{arm}_{cond}_{sample}_steps.json"
                records.append(json.loads(path.read_text()))
            for i in range(64):
                for field in ["uniform_sha256","query_sha256","rng_after"]:
                    assert len({r[i][field] for r in records})==1
            initials={r["initial_state_sha256"] for r in rows if r["seed"]==seed and r["sampling_seed"]==sample}
            assert len(initials)==1
    def iso(a,b):
        check()
        if a.number_of_edges()!=b.number_of_edges():return False,None
        def timeout(sig,frame):raise TimeoutError("isomorphism one-second cap")
        previous=signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,1)
        try:
            gm=nx.algorithms.isomorphism.GraphMatcher(a,b,node_match=lambda x,y:x["type"]==y["type"],
                                                    edge_match=lambda x,y:x["relation"]==y["relation"])
            result=gm.is_isomorphic();return result,gm.mapping if result else None
        except TimeoutError:return None,None
        finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)
    summary=[];mappings=[];uncertain=[]
    metrics=["G_W1","G_joint_TV","G_max","G_high_ge12","G_multiple_hubs","G_zero","G_low_le1"]
    for seed,arm,cond in itertools.product(cfg["train_seeds"],cfg["arms"],["balanced","extreme"]):
        check();rr=[r for r in rows if (r["seed"],r["arm"],r["condition"])==(seed,arm,cond)]
        assert len(rr)==8
        entry=dict(seed=seed,arm=arm,condition=cond,n=8)
        for k in metrics:entry[k]=float(np.mean([r[k] for r in rr]))
        for k in ["edges","lcc","isolates","maximum_degree","triangles"]:
            entry[k]=float(np.mean([r["metrics"][k] for r in rr]))
        cv=[r["metrics"]["imdb_closure"]["path_ratio"] for r in rr]
        entry["closure_NA"]=sum(x is None for x in cv)
        entry["closure_mean"]=float(np.mean([x for x in cv if x is not None])) if any(x is not None for x in cv) else None
        reps=[];freq=[];unknown=0
        for r in rr:
            g=graphs[seed,arm,cond,r["sampling_seed"]];assigned=False
            for j,other in enumerate(reps):
                match,mapping=iso(g,other)
                if match is None:unknown+=1;uncertain.append(dict(seed=seed,arm=arm,condition=cond,sample=r["sampling_seed"]));continue
                if match:
                    freq[j]+=1;assigned=True;mappings.append(dict(seed=seed,arm=arm,condition=cond,
                        sample=r["sampling_seed"],representative_index=j,mapping=mapping));break
            if not assigned:reps.append(g);freq.append(1)
        entry.update(topology_classes=len(reps) if not unknown else None,frequency=freq if not unknown else None,
                     isomorphism_unresolved=unknown,
                     effective_classes=64/sum(n*n for n in freq) if not unknown else None)
        summary.append(entry)
    effects=[]
    for seed,cond in itertools.product(cfg["train_seeds"],["balanced","extreme"]):
        a=next(r for r in summary if (r["seed"],r["arm"],r["condition"])==(seed,"original_C",cond))
        b=next(r for r in summary if (r["seed"],r["arm"],r["condition"])==(seed,"late_C",cond))
        effects.append(dict(seed=seed,condition=cond,late_minus_original={
            k:b[k]-a[k] if a[k] is not None and b[k] is not None else None
            for k in metrics+["edges","lcc","isolates","maximum_degree","triangles","closure_mean","topology_classes"]}))
    condition_changes=[]
    for seed,arm in itertools.product(cfg["train_seeds"],cfg["arms"]):
        changes=0;iso_changes=0;unresolved=0
        for sample in cfg["sampling_seeds"]:
            a=graphs[seed,arm,"balanced",sample];b=graphs[seed,arm,"extreme",sample]
            changes+=set(a.edges)!=set(b.edges)
            match,_=iso(a,b);unresolved+=match is None;iso_changes+=match is False
        condition_changes.append(dict(seed=seed,arm=arm,paired_samples=8,
            labeled_graph_changes=changes,proven_nonisomorphic_changes=iso_changes,unresolved=unresolved))
    save(out/"seed_summary.json",summary);save(out/"paired_effects.json",effects)
    save(out/"topology_mappings.json",mappings);save(out/"condition_generation_response.json",condition_changes)
    # Descriptive plots only; no metric-based checkpoint selection.
    check();import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(10,9))
    for i,seed in enumerate(cfg["train_seeds"]):
        for arm in cfg["arms"]:
            rr=[r for r in updates if r["seed"]==seed and r["arm"]==arm]
            xx=np.array([r["loss"] for r in rr])
            axes[i,0].plot(np.arange(25,1001),np.convolve(xx,np.ones(25)/25,"valid"),label=arm)
            vs=[json.loads((out/f"seed{seed}"/f"{arm}_validation_{u:04d}.json").read_text())["mean_ce"] for u in [0,250,500,1000]]
            axes[i,1].plot([0,250,500,1000],vs,"o-",label=arm)
        axes[i,0].set_title(f"seed {seed}: train CE (25-update mean)")
        axes[i,1].set_title("Fixed same-reference denoising CE")
        for ax in axes[i]:ax.legend();ax.set_xlabel("effective updates")
    fig.tight_layout();fig.savefig(out/"learning_curves.png");plt.close(fig)
    check()
    save(out/"audit.json",dict(status="completed",updates=6000,chains=96,weight_files=12,
         per_graph_checks=checks,initial_common_tensors_exact=True,training_input_RNG_pairs_exact=True,
         sampling_uniform_query_RNG_pairs_exact=True,unresolved=uncertain,
         wall_seconds=time.perf_counter()-started,CPU_seconds=time.process_time()-cpu,
         peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024))
    return 0
if __name__=="__main__":sys.exit(main(sys.argv[1],float(sys.argv[2])))
