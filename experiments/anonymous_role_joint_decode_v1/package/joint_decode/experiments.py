"""Frozen engineering cases. No real model probability or trained weights."""
import itertools
import json
import math
import resource
import time
from pathlib import Path
import numpy as np
from .contracts import CONFIG,Schema,EdgeLaw,canonical_assignment,digest
from .exact import Universe,distributions,draw,metrics,uniform_assignment_counterexample
from .decoder import decode


def homo_case():
    s=Schema(np.zeros(6,dtype=np.int64),[(0,0)])
    cycle=np.array([[0,1,2,3,4,0],[1,2,3,4,5,5]])
    triangles=np.array([[0,0,1,3,3,4],[1,2,2,4,5,5]])
    cycle_bits=np.zeros(s.s,np.uint8);cycle_bits[s.encode_edges(cycle)]=1
    tri_bits=np.zeros(s.s,np.uint8);tri_bits[s.encode_edges(triangles)]=1
    target=s.profile(cycle_bits)
    return s,target,cycle_bits,tri_bits


def hetero_case():
    s=Schema(np.array([0,0,0,1,1,1]),[(0,0),(0,1),(1,1)])
    edges=np.array([[0,3,0,0,2],[1,4,3,4,5]])
    bits=np.zeros(s.s,np.uint8);bits[s.encode_edges(edges)]=1
    # Another actual graph has the SAME one-dimensional histograms but a
    # different joint role combination for type0. This is not a valid target.
    wrong_edges=np.array([[0,3,2,2,0],[1,4,3,4,5]])
    wrong=np.zeros(s.s,np.uint8);wrong[s.encode_edges(wrong_edges)]=1
    return s,s.profile(bits),bits,wrong


def independent_six_census():
    """Separate Python adjacency-set oracle, only C(15,6)=5005 candidates.

    Degree2 forces6 edges. No production bit matrix/degree/connected routine.
    """
    edges=list(itertools.combinations(range(6),2));feasible=[];connected=0;two_tri=0
    for chosen in itertools.combinations(edges,6):
        graph=[set() for _ in range(6)]
        for u,v in chosen:graph[u].add(v);graph[v].add(u)
        if any(len(n)!=2 for n in graph):continue
        reached={0};front=[0]
        while front:
            for v in graph[front.pop()]:
                if v not in reached:reached.add(v);front.append(v)
        feasible.append(chosen)
        if len(reached)==6:connected+=1
        elif len(reached)==3 and all(graph[v]<=reached for v in reached):two_tri+=1
        else:raise AssertionError('degree2 component classification failed')
    return dict(candidates=math.comb(15,6),feasible=len(feasible),six_cycles=connected,two_triangles=two_tri,
        independent_combinatorial_counts=dict(six_cycles=math.factorial(5)//2,two_triangles=math.comb(6,3)//2))


def small(output,check=lambda:None):
    out=Path(output);start=time.monotonic();cpu=time.process_time();cfg_sha=digest(CONFIG)
    s,target,cycle,tri=homo_case();u=Universe(s);mask=u.role_mask(target)
    oracle=independent_six_census()
    if oracle['feasible']!=70 or oracle['six_cycles']!=60 or oracle['two_triangles']!=10:raise AssertionError('independent70/60/10 census failed')
    if int(mask.sum())!=70 or int((mask&u.connected).sum())!=60 or int((mask&~u.connected).sum())!=10:raise AssertionError('production census differs')
    cases=[('all_0.5',u,target,np.full(s.s,.5)),
        ('cycle_0.8_other_0.2',u,target,np.where(cycle,.8,.2)),
        ('two_triangles_0.8_other_0.2',u,target,np.where(tri,.8,.2))]
    hs,ht,href,hwrong=hetero_case();hu=Universe(hs)
    cases.append(('hetero_joint',hu,ht,np.array(CONFIG['hetero_probabilities_canonical_family_slots'])))
    summary=dict(scope='ENGINEERING probabilities, not model outputs',status='incomplete',cases=[],config_sha256=cfg_sha,
        independent_census=oracle,enumerated_homo_graphs=len(u.codes),enumerated_hetero_graphs=len(hu.codes),
        homo_type_preserving_topology_classes=len(np.unique(u.topology)),model_calls=0,GPU_use=0,training_updates=0,scientific_chains=0,
        draws_per_method=CONFIG['draws_per_method'],engineering_graph_draws=len(cases)*5*CONFIG['draws_per_method'],
        exact_metrics='all graph states summed in FP64; only distinct draw counts/observed pass counts are sampling statistics')
    for index,(name,universe,profile,probabilities) in enumerate(cases):
        check();print(json.dumps(dict(case=name,event='enumeration_metrics_start')),flush=True)
        law=EdgeLaw.from_probabilities(probabilities);dist,info,logq,roles=distributions(universe,profile,law)
        rng=np.random.default_rng(np.random.SeedSequence([CONFIG['sampling_seed'],index]))
        uniforms=rng.random(CONFIG['draws_per_method']);methods={};arrays=dict(bits=universe.bits,slots=universe.schema.slots,
            families=universe.schema.families,types=universe.schema.types,role_mask=roles,log_q=logq,canonical_topology=universe.topology,
            engineering_probabilities=probabilities,shared_uniforms=uniforms)
        for method,weights in dist.items():
            check();draws=draw(weights,uniforms)
            support=1 if method=='MAP' else info['distributions'][method]['mathematical_support_size']
            methods[method]=metrics(universe,weights,logq,roles,draws,support)
            arrays[method+'_probabilities']=weights;arrays[method+'_draw_codes']=draws
        assignment=canonical_assignment(universe.schema,profile)
        fixed_bits,solver=decode(universe.schema,profile,assignment,law,seconds=CONFIG['decoder']['per_small_solve_seconds'],check=check)
        fixed_mask=np.all(universe.degrees==assignment[None,:,:],axis=(1,2)) & law.allowed(universe.bits)
        exact_fixed=float(logq[fixed_mask].max())
        if fixed_bits is None or not solver['optimal'] or abs(solver['log_q']-exact_fixed)>2e-7:raise AssertionError('fixed decoder disagrees with exhaustive fixed MAP')
        arrays['fixed_assignment_bits']=fixed_bits;arrays['assigned_joint_degrees']=assignment
        record=dict(name=name,target=profile,probabilities=probabilities.tolist(),truth=info,methods=methods,
            fixed_assignment_MAP=solver,exact_fixed_assignment_log_q=exact_fixed,
            fixed_assignment_graph_code=int(np.sum(fixed_bits.astype(np.uint32)<<np.arange(universe.schema.s))),
            exact_anonymous_MAP_log_q=float(logq[roles].max()),
            role_mixture_counterexample=uniform_assignment_counterexample(universe,dist['conditional_T1'],logq,roles))
        if name=='hetero_joint':
            wrong_code=int(np.sum(hwrong.astype(np.uint32)<<np.arange(hs.s)))
            record['joint_vs_marginals']=dict(joint_feasible_graphs=int(roles.sum()),
                marginal_only_graphs=int(universe.role_mask(profile,True).sum()),
                concrete_wrong_graph_code=wrong_code,wrong_graph_joint_pass=bool(roles[wrong_code]),
                wrong_graph_marginals_pass=bool(universe.role_mask(profile,True)[wrong_code]))
        check();folder=out/name;folder.mkdir();np.savez_compressed(folder/'exact_tables.npz',**arrays)
        (folder/'result.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
        summary['cases'].append(record)
    check()
    if digest(json.loads((Path(__file__).resolve().parents[1]/'frozen_config.json').read_text()))!=cfg_sha:raise ValueError('frozen configuration changed during experiment')
    summary.update(status='complete',wall_seconds=time.monotonic()-start,cpu_seconds=time.process_time()-cpu,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    return summary
