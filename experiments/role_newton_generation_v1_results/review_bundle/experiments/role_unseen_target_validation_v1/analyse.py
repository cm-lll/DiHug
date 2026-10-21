"""Target-only error statistics, independent raw-graph audit and per-seed summaries.

No witness/reference topology is loaded. Structural outputs are descriptive.
"""
from collections import Counter, defaultdict
import itertools
import math
from pathlib import Path
import time
import numpy as np
import networkx as nx
from execute import save, load, digest, PLAN, cfg

NAMES = ["Director", "Actor", "Movie", "Genre"]


def w1(hx, hy):
    n = max(len(hx), len(hy))
    x = np.pad(np.asarray(hx, dtype=float)/sum(hx), (0, n-len(hx)))
    y = np.pad(np.asarray(hy, dtype=float)/sum(hy), (0, n-len(hy)))
    return float(np.abs(np.cumsum(x-y)).sum())


def target_errors(metrics, target):
    result = {}
    for typ, name in enumerate(NAMES):
        rows = target["roles"][typ]
        joint = {",".join(map(str, v)): int(n) for v, n in rows}
        actual = metrics["joint_relation_degree"][name]
        count = target["type_counts"][typ]
        tv = .5*sum(abs(actual.get(k, 0)-joint.get(k, 0))/count for k in actual.keys() | joint.keys())
        errors = []
        for r, a in enumerate(metrics["typed_degree"][name]["by_relation"]):
            hist = np.zeros(max(v[r] for v, n in rows)+1, dtype=int)
            for v, n in rows:hist[v[r]] += n
            errors.append(w1(a["histogram"], hist))
        result[name] = dict(joint_TV=tv, relation_W1=errors)
    g = metrics["typed_degree"]["Genre"]["by_relation"][8]
    values = sorted((v for v, n in enumerate(g["histogram"]) for _ in range(n)), reverse=True)
    result["Genre"].update(W1=errors[8], degrees_descending=values, histogram=g["histogram"],
        max=values[0], second_max=values[1], top_two_sum=values[0]+values[1],
        high_ge8=sum(v>=8 for v in values), high_ge12=sum(v>=12 for v in values),
        multiple_ge8=sum(v>=8 for v in values)>=2, low_le1=sum(v<=1 for v in values), zero=values.count(0))
    sums = []
    for r, (a, b) in enumerate(target["schema"]):
        s = sum(v[r]*n for v, n in target["roles"][a]);sums.append(s//2 if a==b else s)
    result["family_signed_error"] = (np.asarray(metrics["family_edges"])-sums).tolist()
    return result


def independent_graph(path):
    with np.load(path,allow_pickle=False) as z:
        types, edges, families, schema = z["types"], z["edges"], z["relations"]-1, z["schema"]
    assert types.shape==(34,) and edges.ndim==2 and edges.shape[1]==2
    assert len(edges)==len(families) and np.bincount(types).tolist()==[4,6,18,6]
    assert np.array_equal(schema, np.asarray(list(itertools.combinations_with_replacement(range(4),2))))
    G=nx.Graph();G.add_nodes_from((i,dict(type=int(t))) for i,t in enumerate(types))
    d=np.zeros((len(types),len(schema)),dtype=int);counts=np.zeros(len(schema),dtype=int)
    for (u,v),r in zip(edges,families):
        assert 0<=u<v<34 and 0<=r<10 and not G.has_edge(int(u),int(v))
        assert tuple(sorted((int(types[u]),int(types[v]))))==tuple(schema[r])
        assert r in (1,2,5,8)
        G.add_edge(int(u),int(v),relation=int(r));d[u,r]+=1;d[v,r]+=1;counts[r]+=1
    return G,d,counts,types


def independently_check(row, out, target):
    G,d,counts,types=independent_graph(out/row["graph_file"])
    m=row["metrics"]
    assert m["nodes"]==len(G) and m["edges"]==G.number_of_edges()
    assert counts.tolist()==m["family_edges"]
    cc=sorted(map(len,nx.connected_components(G)),reverse=True)
    assert cc==m["component_sizes"] and m["lcc"]==max(cc)
    assert m["connected"]==(len(cc)==1) and m["isolates"]==nx.number_of_isolates(G)
    triangles=sum(nx.triangles(G).values())//3
    assert triangles==m["triangles"]
    typed=Counter(tuple(sorted(int(types[v]) for v in triplet)) for triplet in itertools.combinations(G,3)
                  if all(G.has_edge(u,v) for u,v in itertools.combinations(triplet,2)))
    assert {str(k):v for k,v in sorted(typed.items())}==m["typed_triangles"]
    # Independent combinatorial totals: no production path enumerator calls.
    totals=[G.number_of_edges(),sum(v*(v-1)//2 for _,v in G.degree()),
            sum((G.degree(u)-1)*(G.degree(v)-1)-len(set(G[u])&set(G[v])) for u,v in G.edges())]
    assert totals==[v["total"] for v in m["mp"]]
    for typ,name in enumerate(NAMES):
        joint=Counter(",".join(map(str,x)) for x in d[types==typ])
        assert dict(joint)==m["joint_relation_degree"][name]
        for r in range(10):
            assert np.bincount(d[types==typ,r]).tolist()==m["typed_degree"][name]["by_relation"][r]["histogram"]
    # DMA closure by independent sets of actual length2 paths.
    paths=[(u,mid,v) for mid in np.flatnonzero(types==2) for u in G[mid] for v in G[mid] if types[u]==0 and types[v]==1]
    closed=[p for p in paths if G.has_edge(p[0],p[2])]
    cl=m["imdb_closure"]
    assert [len(closed),len(paths),len({(u,v) for u,_,v in closed}),len({(u,v) for u,_,v in paths})]==[
        cl["path_numerator"],cl["path_denominator"],cl["unique_pair_numerator"],cl["unique_pair_denominator"]]
    err=target_errors(m,target)
    assert err==row["target_errors"]
    actual=sorted(d[types==3,8].tolist());wanted=sorted(v[8] for v,n in target["roles"][3] for _ in range(n))
    assert abs(np.mean(np.abs(np.array(actual)-wanted))-err["Genre"]["W1"])<1e-12
    return G


def stat(values):
    defined=[v for v in values if v is not None]
    return dict(mean=float(np.mean(defined)) if defined else None,
                median=float(np.median(defined)) if defined else None,
                P90=float(np.quantile(defined,.9)) if defined else None,
                defined=len(defined),NA=len(values)-len(defined),values=values)


def features(row):
    m=row["metrics"];e=row["target_errors"]
    f={"G_"+k:e["Genre"][k] for k in ("W1","joint_TV","max","second_max","top_two_sum","high_ge8","high_ge12","multiple_ge8","low_le1","zero")}
    f.update(edges=m["edges"],connected=int(m["connected"]),lcc=m["lcc"],isolates=m["isolates"],
             components=len(m["component_sizes"]),triangles=m["triangles"],clustering=m["average_clustering"],assortativity=m["assortativity"],
             closure_path=m["imdb_closure"]["path_ratio"],closure_pair=m["imdb_closure"]["unique_pair_ratio"])
    for r in range(10):
        f[f"family{r}_signed_error"]=e["family_signed_error"][r]
        f[f"family{r}_absolute_error"]=abs(e["family_signed_error"][r])
        f[f"family{r}_edges"]=m["family_edges"][r]
    for name in NAMES:
        f[name+"_joint_TV"]=e[name]["joint_TV"]
        for r in range(10):f[f"{name}_relation{r}_W1"]=e[name]["relation_W1"][r]
    for i,x in enumerate(m["mp"]):f[f"MP{i+1}_total"]=x["total"]
    return f


def audit(out,check):
    started=time.monotonic();rows=[load(p) for p in sorted(out.glob("seed*/*_result.json"))]
    assert len(rows)==120 and len(list(out.glob("seed*/*_steps.json")))==120
    assert len([p for p in out.glob("seed*/*.npz") if p.name!="partial.npz"])==120
    weights=load(PLAN/"evidence/weights.json")["weights"]
    seeds=[w["seed"] for w in weights]
    targets={r["name"]:r for r in load(PLAN/"evidence/targets.json")["targets"]}
    paired={};groups=defaultdict(list);graphs={};step_seconds=[]
    expected_keys=set(itertools.product(seeds,cfg()["target_order"],range(2026102031,2026102039)))
    assert {(r["seed"],r["condition"],r["sampling_seed"]) for r in rows}==expected_keys
    for r in rows:
        check();target=targets[r["condition"]]
        assert digest(out/r["graph_file"])==r["graph_sha256"] and digest(out/r["steps_file"])==r["steps_sha256"]
        assert r["weight_sha256"]==next(w["sha256"] for w in weights if w["seed"]==r["seed"])
        assert r["target_fingerprint"]==target["fingerprint"] and r["newly_executed_chain"]
        steps=load(out/r["steps_file"])
        assert [s["t"] for s in steps]==list(range(64,0,-1))
        assert all(s["exact_coverage"] and s["blocks"]==4 and s["covered_slots"]==312 for s in steps)
        arr=[(s["t"],s["uniform_sha256"],s["query_sha256"],s["rng_after"]) for s in steps]
        random_binding=(r["initial_state_sha256"],arr)
        if r["sampling_seed"] in paired:assert paired[r["sampling_seed"]]==random_binding
        else:paired[r["sampling_seed"]]=random_binding
        step_seconds += [s["step_seconds"] for s in steps]
        G=independently_check(r,out,target["anonymous_target"])
        graphs[r["graph_file"]]=G
        groups[r["seed"],r["condition"]].append(r)
    summaries=[];classes=[]
    for (seed,condition),group in sorted(groups.items()):
        check();group.sort(key=lambda r:r["sampling_seed"]);assert len(group)==8
        representatives=[];freq=[];classes_for_group=[]
        for row in group:
            check();G=graphs[row["graph_file"]]
            wl=nx.weisfeiler_lehman_graph_hash(G,node_attr="type",edge_attr="relation")
            found=None
            for i,(other,ow) in enumerate(representatives):
                if wl==ow and nx.is_isomorphic(G,other,node_match=lambda x,y:x["type"]==y["type"],edge_match=lambda x,y:x["relation"]==y["relation"]):
                    found=i;break
            if found is None:found=len(representatives);representatives.append((G,wl));freq.append(0)
            freq[found]+=1;classes_for_group.append(dict(sampling_seed=row["sampling_seed"],local_class=found))
        ff=[features(r) for r in group]
        summaries.append(dict(seed=seed,condition=condition,graphs=8,
            statistics={k:stat([x[k] for x in ff]) for k in ff[0]},
            mean_G_histogram=np.mean([np.bincount(r["target_errors"]["Genre"]["degrees_descending"],minlength=19) for r in group],axis=0).tolist(),
            topology_classes=len(freq),topology_frequencies=freq,effective_classes=math.exp(-sum(n/8*math.log(n/8) for n in freq)),
            inverse_simpson=1/sum((n/8)**2 for n in freq)))
        classes.append(dict(seed=seed,condition=condition,frequencies=freq,assignments=classes_for_group))
    # Same-sampling-seed condition differences within each fixed model, kept as paired draws.
    differences=[]
    for seed in seeds:
        for name in cfg()["target_order"][1:]:
            left=sorted(groups[seed,"balanced"],key=lambda r:r["sampling_seed"])
            right=sorted(groups[seed,name],key=lambda r:r["sampling_seed"])
            a,b=[features(r) for r in left],[features(r) for r in right]
            differences.append(dict(seed=seed,condition=name,relative_to="balanced",
                statistics={k:stat([y[k]-x[k] if x[k] is not None and y[k] is not None else None for x,y in zip(a,b)]) for k in a[0]},
                G_W1_note="each condition's own target; signed difference is descriptive, not error against one common reference"))
    check();save(out/"generation.json",rows)
    save(out/"per_seed.json",dict(summaries=summaries,paired_condition_differences=differences,
        unit="three training seeds; each cell eight paired draws; no pooling as independent training replicates"))
    save(out/"topology_classes.json",classes)
    save(out/"audit.json",dict(status="passed",graphs=120,reverse_steps=7680,network_forwards=30720,
        wall_seconds=time.monotonic()-started,step_seconds=stat(step_seconds),
        checks=["all120 newly executed complete chains", "weight/target/source binding", "all7680 active-slot full sweeps",
                "identical paired prior/query/uniform/RNG arrangements", "independent graph schema/degrees/joint frequencies/W1",
                "independent connectivity/typed triangles/path totals/DMA closure", "exact typed topology classes within each cell"],
        not_independently_covered=["network probabilities not recomputed by another implementation", "MP pattern counts reuse frozen verified enumerator; totals independently checked"],
        training_updates=0,GPU_used=False,proof_graphs_used=False))
