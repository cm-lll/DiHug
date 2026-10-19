"""One <=300s fixed-matching engineering capacity test, NOT model evaluation."""
import hashlib
import itertools
import json
import time
import resource
from pathlib import Path
import numpy as np
from .contracts import CONFIG,Schema,EdgeLaw,canonical_assignment,digest
from .decoder import decode


def engineering_logits(schema):
    # Signature deliberately has no reference edges, target or role assignment.
    rng=np.random.default_rng(CONFIG['imdb_engineering']['score_seed'])
    return rng.standard_normal(schema.s)


def run(output,deadline,check):
    out=Path(output);start=time.monotonic();cpu=time.process_time()
    reference=Path(__file__).resolve().parents[1]/'data/IMDB_movie_reference.npz'
    sha=hashlib.sha256(reference.read_bytes()).hexdigest()
    if sha!=CONFIG['imdb_engineering']['reference_sha256']:raise ValueError('public full IMDB reference hash differs')
    with np.load(reference,allow_pickle=False) as z:
        types=z['types'];edges=z['edges'].T;relations=z['relations']-1
    schema=Schema(types,np.array(list(itertools.combinations_with_replacement(range(4),2))))
    ids=schema.encode_edges(edges)
    if not np.array_equal(schema.families[ids],relations):raise ValueError('reference relation remapping differs')
    refbits=np.zeros(schema.s,np.uint8);refbits[ids]=1
    target=schema.profile(refbits);assigned=canonical_assignment(schema,target)
    # No later step reads the actual reference edge identities.
    del edges,relations,ids,refbits
    artifact=dict(profile=target,reference_sha256=sha,definition_baseline=CONFIG['baseline_definition_commit'])
    (out/'anonymous_target.json').write_text(json.dumps(artifact,indent=2)+'\n')
    check();tick=time.monotonic();law=EdgeLaw(engineering_logits(schema));score_seconds=time.monotonic()-tick
    events=[]
    def event(row):
        check();events.append(row);print(json.dumps(dict(event='relation_done',**row)),flush=True)
    remaining=deadline-time.monotonic()
    if remaining<=0:raise TimeoutError('no decoder budget after preprocessing')
    bits,report=decode(schema,target,assigned,law,seconds=remaining,check=check,event=event)
    result=dict(status='incomplete',scope='IMDB-size ENGINEERING fixed-matching decoder ONLY; NO model probabilities',
        config_sha256=digest(CONFIG),reference_sha256=sha,target_sha256=digest(artifact),
        nodes=schema.n,slots=schema.s,type_counts=schema.sizes,
        roles={c:dict(nodes=g['nodes'],distinct_vectors=len(g['rows'])) for c,g in target['groups'].items()},
        engineering_score_seed=CONFIG['imdb_engineering']['score_seed'],score_generation_seconds=score_seconds,
        score_independent_of_reference_edge_identity=True,decoder=report,model_probabilities_available=False,
        model_calls=0,training_updates=0,GPU_use=0,scientific_chains=0,
        no_anonymous_global_MAP_claim=True,no_conditional_sampling_claim=True)
    if bits is not None:
        check();selected=schema.slots[:,bits.astype(bool)]
        graph=out/'engineering_graph.npz'
        np.savez_compressed(graph,types=schema.types,edges=selected,families=schema.families[bits.astype(bool)],
                            assigned_joint_degrees=assigned,actual_joint_degrees=schema.degree(bits))
        check();result.update(status='complete',graph_sha256=hashlib.sha256(graph.read_bytes()).hexdigest(),
            engineering_graphs=1,physical_edges=int(bits.sum()),
            relation_edges=np.bincount(schema.families,weights=bits,minlength=schema.r).astype(int).tolist())
    elif report['status'] not in ('incomplete',):result['status']='failed'
    check()
    if hashlib.sha256(reference.read_bytes()).hexdigest()!=sha:raise ValueError('reference mutated during run')
    result.update(wall_seconds=time.monotonic()-start,cpu_seconds=time.process_time()-cpu,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    return result
