"""Frozen PR0 cycle templates; reference permission is scalar summaries only."""
from itertools import combinations


def _fast_kind(template):
    t=tuple(template)
    if len(t)==3 and len(set(t)) in (1,3):return 'triangle'
    if len(t)==4 and t[0]==t[2] and t[1]==t[3] and t[0]!=t[1]:return 'bipartite4'
    return None


def _typed_neighbors(graph,node,typename):
    return {v for v in graph.adjacency[node] if graph.type_names[int(graph.node_types[v])]==typename}


def _affected_instances(graph,template,affected_edges):
    """Same physical cycles as DFS; prune by the already fixed type template."""
    kind=_fast_kind(template);found=set();allowed=set(template)
    for u,v in affected_edges:
        if not graph.has_edge(u,v):continue
        a=graph.type_names[int(graph.node_types[u])];b=graph.type_names[int(graph.node_types[v])]
        if a not in allowed or b not in allowed:continue
        if kind=='triangle':
            if len(allowed)==1:
                if a!=b:continue
                third=a
            else:
                if a==b:continue
                third=next(iter(allowed-{a,b}))
            for w in graph.adjacency[u]&graph.adjacency[v]:
                if graph.type_names[int(graph.node_types[w])]==third:found.add(_key((u,v,w)))
        else:
            if a==b:continue
            for w in _typed_neighbors(graph,v,a)-{u}:
                for x in (graph.adjacency[u]&graph.adjacency[w])-{v}:
                    if graph.type_names[int(graph.node_types[x])]==b:found.add(_key((u,v,w,x)))
    return found


def _fast_count(graph,template):
    kind=_fast_kind(template);t=tuple(template)
    groups={name:[u for u in range(len(graph.node_types)) if graph.type_names[int(graph.node_types[u])]==name] for name in set(t)}
    if kind=='triangle':
        if len(groups)==1:
            total=0
            for u in groups[t[0]]:
                for v in _typed_neighbors(graph,u,t[0]):
                    if v>u:total+=sum(w>v for w in _typed_neighbors(graph,u,t[0])&_typed_neighbors(graph,v,t[0]))
            return total
        return sum(len(_typed_neighbors(graph,u,t[2])&_typed_neighbors(graph,v,t[2]))
                   for u in groups[t[0]] for v in _typed_neighbors(graph,u,t[1]))
    from collections import Counter
    left={u:_typed_neighbors(graph,u,t[1]) for u in groups[t[0]]}
    right={u:_typed_neighbors(graph,u,t[0]) for u in groups[t[1]]}
    side=min((left,right),key=lambda x:sum(len(v)*(len(v)-1)//2 for v in x.values()))
    shared=Counter()
    for ns in side.values():shared.update(combinations(sorted(ns),2))
    return sum(c*(c-1)//2 for c in shared.values())


def _match(types, template):
    return any(tuple(types[i:]+types[:i]) == tuple(template) or
               tuple(list(reversed(types))[i:]+list(reversed(types))[:i]) == tuple(template)
               for i in range(len(types)))


def _key(path):
    path = tuple(path)
    rev = tuple(reversed(path))
    return min([p[i:]+p[:i] for p in (path, rev) for i in range(len(p))])


def cycle_instances(graph, template, affected_edges=None):
    """Sparse DFS around affected edges, with canonical cycle deduplication.

    A four-node set can contain three different cycles; canonicalize the
    cycle edge order, not just the unordered node set. Non-induced cycles.
    """
    template = tuple(template)
    if len(template) not in (3, 4):
        raise ValueError('only frozen length-three/four templates are supported')
    if any(t not in graph.type_names for t in template):
        raise ValueError('motif type absent from schema')
    if affected_edges is not None and _fast_kind(template):
        return _affected_instances(graph,template,affected_edges)
    starts = (graph.edge_family.keys() if affected_edges is None else affected_edges)
    found = set()
    for u, v in starts:
        if not graph.has_edge(u, v):
            continue
        def visit(path):
            if len(path) == len(template):
                if graph.has_edge(path[-1], path[0]):
                    types = [graph.type_names[int(graph.node_types[n])] for n in path]
                    if _match(types, template): found.add(_key(path))
                return
            for n in graph.adjacency[path[-1]]:
                if n not in path: visit(path+[n])
        visit([u, v])
    return found


def motif_counts(graph, templates):
    if any(name not in graph.type_names for t in templates for name in t):
        raise ValueError('motif type absent from schema')
    return tuple(_fast_count(graph,t) if _fast_kind(t) else len(cycle_instances(graph,t)) for t in templates)


def motif_delta(graph, proposal, templates):
    affected = set(graph.canonical_edge(*e) for e in proposal.old_edges+proposal.new_edges)
    if not affected: return (0,)*len(templates)
    # Overlay only edited endpoint neighborhoods; no all-graph clone/head.
    changed = {n for e in proposal.old_edges+proposal.new_edges for n in e}
    neighbors = {n:set(graph.adjacency[n]) for n in changed}
    for u,v in proposal.old_edges:
        neighbors[u].remove(v); neighbors[v].remove(u)
    for u,v in proposal.new_edges:
        neighbors[u].add(v); neighbors[v].add(u)
    class Adjacency:
        def __getitem__(self,n): return neighbors[n] if n in neighbors else graph.adjacency[n]
    class Virtual:
        adjacency = Adjacency()
        node_types,type_names = graph.node_types,graph.type_names
        def has_edge(self,u,v): return v in self.adjacency[u]
    edited = Virtual()
    return tuple(len(cycle_instances(edited, t, affected))-len(cycle_instances(graph, t, affected))
                 for t in templates)


def motif_loss(counts, reference_counts):
    if len(counts) != 2 or len(reference_counts) != 2 or any(int(x) != x or x < 0 for x in reference_counts):
        raise ValueError('two nonnegative integer public motif counts required')
    return sum(abs(a-b)/max(1, b) for a,b in zip(counts, reference_counts))/2


def rule_scores(graph, candidates, templates, reference_counts):
    cache=getattr(graph,'_pr3_motif_cache',None)
    current = cache['counts'] if cache is not None and cache['templates']==tuple(map(tuple,templates)) else motif_counts(graph, templates)
    before = motif_loss(current, reference_counts)
    deltas=[motif_delta(graph,p,templates) for p in candidates]
    if cache is not None:cache['candidate_deltas']=deltas
    return [before-motif_loss(tuple(a+b for a,b in zip(current,delta)),reference_counts) for delta in deltas]
