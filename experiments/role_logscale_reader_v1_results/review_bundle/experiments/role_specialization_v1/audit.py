"""CPU independent final-graph/paired-log audit within the original new ledger."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import resource
import statistics
import time
import numpy as np
import networkx as nx

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def graph_npz(p):
    with np.load(p,allow_pickle=False) as z:
        types=z["types"];edges=z["edges"];families=z["relations"]-1;schema=z["schema"]
    assert edges.shape[1]==2
    G=nx.Graph()
    G.add_nodes_from((i,{"type":int(t)}) for i,t in enumerate(types))
    deg=np.zeros((len(types),len(schema)),dtype=int);counts=np.zeros(len(schema),dtype=int)
    for (u,v),r in zip(edges,families):
        assert 0<=u<v<len(types) and not G.has_edge(int(u),int(v))
        assert tuple(sorted((int(types[u]),int(types[v]))))==tuple(schema[r])
        G.add_edge(int(u),int(v),relation=int(r));deg[u,r]+=1;deg[v,r]+=1;counts[r]+=1
    return G,deg,counts,types
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--run",type=Path,required=True);ap.add_argument("--source",type=Path,required=True)
    a=ap.parse_args();started=time.monotonic();cpu=time.process_time()
    ledger=json.loads((a.run/"activation.json").read_text())
    def check():
        if time.time()>ledger["deadline_epoch"] or (a.run/"STOP").exists():raise TimeoutError("original panel/audit deadline")
    check();assert json.loads((a.run/"worker_receipt.json").read_text())["status"]=="completed"
    updates=[json.loads(x) for x in (a.run/"updates.jsonl").read_text().splitlines()]
    assert len(updates)==3000
    old={(r["seed"],r["update"]):r for r in map(json.loads,(a.source/"updates.jsonl").read_text().splitlines()) if r["arm"]=="late_C"}
    for seed in [2026102011,2026102012,2026102013]:
        rr=[r for r in updates if r["seed"]==seed]
        assert [r["update"] for r in rr]==list(range(1,1001))
        assert all(r["input_sha256"]==old[seed,r["update"]]["input_sha256"] and r["optimizer_step"]==r["update"] for r in rr)
    weights=json.loads((a.run/"weight_hashes.json").read_text());assert len(weights)==6
    for p,h in weights.items():check();assert sha(a.run/p)==h
    new=json.loads((a.run/"generation.json").read_text());assert len(new)==48
    before=[r for r in json.loads((a.source/"generation.json").read_text()) if r["arm"]=="late_C"]
    graphs={};independent=[]
    for r in new:
        check();path=a.run/r["graph_file"];assert sha(path)==r["graph_file_sha256"]
        G,d,counts,types=graph_npz(path);met=r["metrics"]
        assert counts.tolist()==met["family_edges"]
        assert sorted(map(len,nx.connected_components(G)),reverse=True)==met["component_sizes"]
        triangles=sum(G.has_edge(i,j) and G.has_edge(i,k) and G.has_edge(j,k) for i,j,k in itertools.combinations(G.nodes,3))
        assert triangles==met["triangles"]
        mg=8;gd=d[types==3,mg];assert gd.max()==r["G_max"] and np.sum(gd>=12)==r["G_high_ge12"]
        target=np.array([4]*6 if r["condition"]=="balanced" else [18,2,1,1,1,1])
        w1=float(np.abs(np.sort(gd)-np.sort(target)).mean())
        assert abs(w1-r["G_W1"])<2e-12
        num=den=0
        for m in np.flatnonzero(types==2):
            ds=[v for v in G.neighbors(int(m)) if types[v]==0];actors=[v for v in G.neighbors(int(m)) if types[v]==1]
            for di,ac in itertools.product(ds,actors):den+=1;num+=int(G.has_edge(di,ac))
        assert (num,den)==(met["imdb_closure"]["path_numerator"],met["imdb_closure"]["path_denominator"])
        old_r=next(x for x in before if (x["seed"],x["condition"],x["sampling_seed"])==(r["seed"],r["condition"],r["sampling_seed"]))
        ns=json.loads(path.with_name(path.stem+"_steps.json").read_text())
        op=a.source/old_r["graph_file"];bs=json.loads(op.with_name(op.stem+"_steps.json").read_text())
        assert len(ns)==64
        for n,b in zip(ns,bs):
            assert n["t"]==b["t"] and n["blocks"]==4 and n["exact_coverage"]
            for key in ["uniform_sha256","query_sha256","rng_after"]:assert n[key]==b[key]
        graphs[r["seed"],r["condition"],r["sampling_seed"]]=G
        independent.append(dict(file=r["graph_file"],W1_abs_error=abs(w1-r["G_W1"]),triangle_exact=True,degree_edges_components_closure_exact=True))
    summaries=[]
    for seed in [2026102011,2026102012,2026102013]:
        for condition in ["balanced","extreme"]:
            for label,records in [("late_C",before),("capacity_C",new)]:
                rr=[r for r in records if r["seed"]==seed and r["condition"]==condition];assert len(rr)==8
                row=dict(seed=seed,arm=label,condition=condition,n=8)
                for key in ["G_W1","G_joint_TV","G_max","G_high_ge12","G_multiple_hubs","G_zero","G_low_le1"]:
                    row[key]=statistics.mean(float(r[key]) for r in rr)
                for key in ["edges","lcc","isolates","triangles"]:
                    row[key]=statistics.mean(r["metrics"][key] for r in rr)
                row["connected"]=sum(r["metrics"]["connected"] for r in rr)
                closure=[r["metrics"]["imdb_closure"]["path_ratio"] for r in rr]
                row["closure_NA"]=sum(v is None for v in closure)
                row["closure"]=None if any(v is None for v in closure) else statistics.mean(closure)
                row["edge_MAE"]=statistics.mean(abs(r["metrics"]["edges"]-88) for r in rr)
                summaries.append(row)
    # Exact isomorphism on these34node graphs; budget check between each pair.
    topology=[];switch=[]
    node=nx.algorithms.isomorphism.categorical_node_match("type",-1)
    edge=nx.algorithms.isomorphism.categorical_edge_match("relation",-1)
    for seed in [2026102011,2026102012,2026102013]:
        for condition in ["balanced","extreme"]:
            reps=[];freq=[]
            for sample_seed in range(2026102031,2026102039):
                check();G=graphs[seed,condition,sample_seed];found=False
                for i,H in enumerate(reps):
                    check()
                    if nx.is_isomorphic(G,H,node_match=node,edge_match=edge):freq[i]+=1;found=True;break
                if not found:reps.append(G);freq.append(1)
            topology.append(dict(seed=seed,condition=condition,classes=len(reps),frequency=freq,
                effective_classes=64/sum(n*n for n in freq)))
        changed=sum(set(graphs[seed,"balanced",ss].edges)!=set(graphs[seed,"extreme",ss].edges) for ss in range(2026102031,2026102039))
        switch.append(dict(seed=seed,changed_graph_pairs=changed,total=8))
    check()
    for p,h in json.loads((a.run/"baseline_sources.json").read_text()).items():assert sha(Path(p))==h
    (a.run/"seed_summary.json").write_text(json.dumps(summaries,indent=2)+"\n")
    (a.run/"topology.json").write_text(json.dumps(topology,indent=2)+"\n")
    (a.run/"condition_switch.json").write_text(json.dumps(switch,indent=2)+"\n")
    (a.run/"audit.json").write_text(json.dumps(dict(status="completed",updates=3000,chains=48,
        baseline_chains_reused=48,weights=6,paired_inputs_uniforms_queries_verified=True,
        graphs=independent,wall_seconds=time.monotonic()-started,CPU_seconds=time.process_time()-cpu,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        cumulative_seconds=time.time()-ledger["activation_epoch"],budget_seconds=2700,no_reset=True),indent=2)+"\n")
    print(json.dumps(switch));print("audit completed")
if __name__=="__main__":main()
