"""Deterministic graph diagnostics; never a loss, constraint or filter."""
import itertools
import numpy as np


def evaluate(types,edges,families,pairs,role_types=None):
    types=np.asarray(types);edges=np.asarray(edges).reshape(2,-1);families=np.asarray(families)
    adj=[{} for _ in types]
    for (u,v),r in zip(edges.T,families):
        if u==v or v in adj[u]:raise ValueError("non-simple graph in evaluator")
        if tuple(sorted((int(types[u]),int(types[v]))))!=tuple(pairs[r]):raise ValueError("schema mismatch")
        adj[u][v]=int(r);adj[v][u]=int(r)
    triangles={};total=0
    # Independent set intersections, no production wedge extraction reused.
    for u in range(len(types)):
        for v in adj[u]:
            if v<=u:continue
            for w in adj[u].keys()&adj[v].keys():
                if w<=v:continue
                vertices=(u,v,w)
                patterns=[]
                for a,b,c in itertools.permutations(vertices):
                    patterns.append((int(types[a]),adj[a][b],int(types[b]),adj[b][c],int(types[c]),adj[c][a]))
                key=",".join(map(str,min(patterns)))
                triangles[key]=triangles.get(key,0)+1;total+=1
    closure=None
    if role_types is not None:
        dt,mt,at=role_types
        open_paths,closed=0,0
        for m in np.flatnonzero(types==mt):
            ds=[u for u in adj[m] if types[u]==dt]
            actors=[v for v in adj[m] if types[v]==at]
            open_paths+=len(ds)*len(actors)
            closed+=sum(a in adj[d] for d in ds for a in actors)
        closure=dict(definition="Director-Movie-Actor simple path instances whose Director-Actor edge exists",
            paths=open_paths,closed_paths=closed,rate=closed/open_paths if open_paths else None,
            role_types=list(role_types),zero_denominator="NA; no pseudocount")
    return dict(triangles=total,typed_relation_triangle_patterns=triangles,director_movie_actor_closure=closure,
        scope="descriptive diagnostics only; no acceptance threshold or graph repair")


def imdb_roles(provenance):
    meta=provenance["source_metadata"] if "source_metadata" in provenance else provenance.get("metadata",{})
    # Bound full preprocessing record supplies the original insertion/type order.
    def find(obj):
        if isinstance(obj,dict):
            if "num_nodes_by_type" in obj:return list(obj["num_nodes_by_type"])
            for value in obj.values():
                result=find(value)
                if result:return result
        return None
    names=find(provenance)
    if names!=["Director","Actor","Movie","Genre"]:
        raise RuntimeError("IMDB role/type binding not verified")
    return (names.index("Director"),names.index("Movie"),names.index("Actor"))
