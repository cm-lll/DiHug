"""All-sample, per-training-seed audit. No model or sampling calls."""
from collections import Counter,defaultdict
import json
from pathlib import Path
import time
import networkx as nx
import numpy as np
from core import fingerprint,load_panel,save,torch

def read_graph(path):
    with np.load(path,allow_pickle=False) as z:
        types=z["types"];edges=z["edges"];families=z["relations"]-1;pairs=z["schema"]
    G=nx.Graph();G.add_nodes_from((i,dict(type=int(t))) for i,t in enumerate(types))
    degree=np.zeros((len(types),len(pairs)),dtype=int);counts=np.zeros(len(pairs),dtype=int)
    for (u,v),f in zip(edges,families):
        assert 0<=u<v<len(types) and not G.has_edge(int(u),int(v))
        assert tuple(sorted((int(types[u]),int(types[v]))))==tuple(pairs[f])
        G.add_edge(int(u),int(v),relation=int(f));degree[u,f]+=1;degree[v,f]+=1;counts[f]+=1
    return G,degree,counts,types

def iso(a,b):return nx.is_isomorphic(a,b,node_match=lambda x,y:x["type"]==y["type"],edge_match=lambda x,y:x["relation"]==y["relation"])

def audit(out,cfg,check):
    start=time.monotonic();updates=[json.loads(x) for x in (out/"updates.jsonl").read_text().splitlines()]
    generations=json.loads((out/"generation.json").read_text())
    assert len(updates)==cfg["expected_updates"] and len(generations)==cfg["expected_chains"]
    weight_hashes=json.loads((out/"weight_hashes.json").read_text());assert len(weight_hashes)==12
    for name,h in weight_hashes.items():check();assert fingerprint(out/name)==h
    update_groups=defaultdict(list)
    for r in updates:
        assert r["forward_count"]==2 and r["optimizer_updates"]==1
        assert abs(r["loss"]-.5*sum(p["loss"] for p in r["parts"]))<1e-12
        update_groups[r["seed"],r["arm"]].append(r)
    for seed in cfg["train_seeds"]:
        for arm in cfg["arms"]:assert [r["update"] for r in update_groups[seed,arm]]==list(range(1,1001))
        left,right=[torch.load(out/f"seed{seed}/{arm}_initial.pt",map_location="cpu",weights_only=True) for arm in cfg["arms"]]
        assert left.keys()==right.keys() and all(torch.equal(left[k],right[k]) for k in left)
        for l,r in zip(update_groups[seed,"normal2"],update_groups[seed,"robust"]):
            assert l["ordinary_input_sha256"]==r["ordinary_input_sha256"]
            assert l["perturbed_input_sha256"]==r["perturbed_input_sha256"] and l["corruption"]==r["corruption"]
            for ll,rr in zip(l["parts"],r["parts"]):
                assert (ll["t"],ll["query_block"],ll["query_count"])==(rr["t"],rr["query_block"],rr["query_count"])
    groups=defaultdict(list);objects={};steps={}
    for r in generations:
        check();path=out/r["graph_file"];assert fingerprint(path)==r["graph_file_sha256"]
        G,d,counts,types=read_graph(path);met=r["metrics"]
        assert met["family_edges"]==counts.tolist()
        assert sorted(map(len,nx.connected_components(G)),reverse=True)==met["component_sizes"]
        assert sum(nx.triangles(G).values())//3==met["triangles"]
        gd=d[types==3,8];target=np.array([4]*6 if r["condition"]=="balanced" else [18,2,1,1,1,1])
        assert abs(np.abs(np.sort(gd)-np.sort(target)).mean()-r["G_W1"])<1e-12
        assert r["G_max"]==int(gd.max()) and r["G_high_ge12"]==int((gd>=12).sum())
        for typ,name in enumerate(("Director","Actor","Movie","Genre")):
            joint=Counter(tuple(map(int,x)) for x in d[types==typ])
            stored={tuple(map(int,k.split(","))):v for k,v in met["joint_relation_degree"][name].items()}
            assert dict(joint)==stored
        key=(r["seed"],r["arm"],r["condition"]);groups[key].append(r);objects[r["graph_file"]]=G
        step=json.loads(path.with_name(path.stem+"_steps.json").read_text());assert [s["t"] for s in step]==list(range(64,0,-1))
        assert all(s["exact_coverage"] for s in step);steps[r["seed"],r["arm"],r["condition"],r["sampling_seed"]]=step
    for seed in cfg["train_seeds"]:
        for cond in ("balanced","extreme"):
            for sample in cfg["sampling_seeds"]:
                l,r=(steps[seed,arm,cond,sample] for arm in cfg["arms"])
                for a,b in zip(l,r):
                    for k in ("t","uniform_sha256","query_sha256","rng_after"):assert a[k]==b[k]
    summaries=[]
    for key,rows in sorted(groups.items()):
        assert len(rows)==8;classes=[];freq=[]
        for r in rows:
            check();G=objects[r["graph_file"]]
            # WL only screens unequal graphs; exact typed isomorphism certifies equality.
            h=nx.weisfeiler_lehman_graph_hash(G,node_attr="type",edge_attr="relation")
            found=None
            for j,(hh,H) in enumerate(classes):
                check()
                if h==hh and iso(G,H):found=j;break
            if found is None:classes.append((h,G));freq.append(1)
            else:freq[found]+=1
        def mean(f):return float(np.mean([f(r) for r in rows]))
        def closure(key):
            values=[r["metrics"]["imdb_closure"][key] for r in rows]
            return float(np.mean(values)) if all(v is not None for v in values) else None
        mg_bias=mean(lambda r:r["comparison"]["family_edge_difference"][8])
        summaries.append(dict(seed=key[0],arm=key[1],condition=key[2],graphs=8,
            G_W1=mean(lambda r:r["G_W1"]),G_joint_TV=mean(lambda r:r["G_joint_TV"]),G_max=mean(lambda r:r["G_max"]),
            G_high_ge12=mean(lambda r:r["G_high_ge12"]),G_multiple_hubs=mean(lambda r:r["G_multiple_hubs"]),G_low_le1=mean(lambda r:r["G_low_le1"]),
            G_zero=mean(lambda r:r["G_zero"]),MG_signed_bias=mg_bias,MG_absolute_error=mean(lambda r:abs(r["comparison"]["family_edge_difference"][8])),
            family_signed_bias=np.mean([r["comparison"]["family_edge_difference"] for r in rows],axis=0).tolist(),
            family_absolute_error=np.mean([np.abs(r["comparison"]["family_edge_difference"]) for r in rows],axis=0).tolist(),
            edges=mean(lambda r:r["metrics"]["edges"]),connected=mean(lambda r:r["metrics"]["connected"]),
            lcc=mean(lambda r:r["metrics"]["lcc"]),isolates=mean(lambda r:r["metrics"]["isolates"]),triangles=mean(lambda r:r["metrics"]["triangles"]),
            closure_path=closure("path_ratio"),closure_pair=closure("unique_pair_ratio"),
            closure_path_defined=sum(r["metrics"]["imdb_closure"]["path_ratio"] is not None for r in rows),
            closure_pair_defined=sum(r["metrics"]["imdb_closure"]["unique_pair_ratio"] is not None for r in rows),
            topology_classes=len(freq),topology_frequencies=freq,inverse_simpson=1/sum((f/8)**2 for f in freq)))
    differences=[]
    fields=("G_W1","G_joint_TV","G_max","G_high_ge12","G_multiple_hubs","G_low_le1","G_zero","MG_signed_bias","MG_absolute_error","edges","connected","lcc","isolates","triangles","closure_path","closure_pair")
    for seed in cfg["train_seeds"]:
        for cond in ("balanced","extreme"):
            l=next(r for r in summaries if (r["seed"],r["condition"],r["arm"])==(seed,cond,"normal2"))
            r=next(r for r in summaries if (r["seed"],r["condition"],r["arm"])==(seed,cond,"robust"))
            differences.append(dict(seed=seed,condition=cond,**{f:r[f]-l[f] if r[f] is not None and l[f] is not None else None for f in fields}))
    save(out/"paired_results.json",dict(per_seed=summaries,robust_minus_normal2=differences,
        scope="eight graphs first summarized within each of three independent training seeds; descriptive development results"))
    save(out/"audit.json",dict(status="passed",updates=len(updates),graphs=len(generations),wall_seconds=time.monotonic()-start,
        checks=["initial tensors and hashes","all two-input paired updates","full64step RNG/query contracts","raw schema/degrees/joint frequencies","independent connectivity/triangles/W1","exact typed topology classes"]))
