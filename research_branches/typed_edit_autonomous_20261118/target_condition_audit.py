"""Finite target-feasibility check; witness graphs are never generation outputs."""
import hashlib
import networkx as nx
import confirmation_common as contract
from round01 import t,b,np,torch,time,Path,ROOT,SRC,json
OUT=ROOT/'results/typed_edit_autonomous_20261118/target_condition_audit'
SEEDS=[100,101,13001,13002]

def feasible(ref,target):
    records={}
    for j,f in enumerate(sorted(ref.family_specs)):
        spec=ref.family_specs[f];left=target[np.array(spec.src_nodes),j]
        if spec.bipartite:
            right=target[np.array(spec.dst_nodes),j];aa=np.sort(left)[::-1];sums=np.cumsum(aa)
            ok=int(left.sum())==int(right.sum()) and all(int(sums[k-1])<=int(np.minimum(right,k).sum()) for k in range(1,len(left)+1)) and bool((left>=0).all() and (right>=0).all())
        else:ok=nx.is_graphical(left.tolist(),method='eg')
        records[f]=bool(ok)
    return records

def main():
    b.setup();OUT.mkdir(parents=True,exist_ok=False);rows=[];start=time.monotonic()
    files=[Path(__file__)]+[b.RUN/f'data/{ds}/graph.json' for ds in ['acm','freebase']]
    contract.bind(OUT,files,read_only_condition_audit=True,initialization_seeds=SEEDS,new_fits=0,new_generated_surrogates=0)
    for ds in ['acm','freebase']:
        ref=b.graph(ds);role=b.ExchangeableRoleDegreePrior.from_training_graph(ref);prior=b.MarginalGraphPrior.from_training_graph(ref);condition=b.FamilyCountCondition(prior,role.family_counts(prior.family_endpoints));names=sorted(ref.family_specs)
        assert len({tuple(sorted((s.src_type,s.dst_type))) for s in ref.family_specs.values()})==len(names)
        for spec in ref.family_specs.values():
            assert set(spec.src_nodes)==set(np.flatnonzero(ref.node_types==spec.src_type))
            assert set(spec.dst_nodes)==set(np.flatnonzero(ref.node_types==spec.dst_type))
        for seed in SEEDS:
            for mode in ['binomial','fixed']:
                assert time.monotonic()-start<180
                initial=condition.sample_graph(seed+71,mode=mode);target=role.sample_assignment_for_graph(initial,seed+91919);unadjusted=role.sample_assignment(initial.node_types,seed+91919);checks=feasible(ref,target);witness_verified=False
                if mode=='fixed':
                    assert np.array_equal(target,unadjusted);rng=np.random.default_rng(seed+91919);mapping=np.empty(ref.num_nodes,dtype=np.int64)
                    for type_id in range(len(ref.type_names)):
                        nodes=np.flatnonzero(ref.node_types==type_id);permutation=rng.permutation(len(nodes));mapping[nodes[permutation]]=nodes
                    witness=b.TypedGraph(ref.node_types.copy(),ref.type_names,ref.family_specs,{f:[tuple(sorted((int(mapping[u]),int(mapping[v])))) for u,v in ref.family_edges[f]] for f in names});witness.validate()
                    degrees=np.column_stack([witness.family_degrees(f) for f in names]);assert np.array_equal(degrees,target) and all(checks.values());witness_verified=True
                rows.append(dict(dataset=ds,initialization_seed=seed,count_mode=mode,edge_count=sum(len(v) for v in initial.family_edges.values()),reference_edge_count=sum(len(v) for v in ref.family_edges.values()),degree_entries_changed_by_apportionment=int(np.count_nonzero(target!=unadjusted)),exact_target_feasible=all(checks.values()),family_checks=checks,permuted_reference_witness_verified=witness_verified,witness_used_for_generation=False,target_sha256=hashlib.sha256(target.tobytes()).hexdigest()))
    contract.verify(OUT);contract.put(OUT,'summary.json',dict(rows=rows,scope='Finite engineering check plus constructive witness for fixed-count type-permutation targets; not a stage-one arrival or learning result.',new_fits=0,new_generated_surrogates=0,seconds=time.monotonic()-start));contract.put(OUT,'acceptance.json',dict(passed=True,target_assignments=16,fixed_count_witnesses=8,new_fits=0,new_generated_surrogates=0));print(json.dumps([{k:v for k,v in row.items() if k!='family_checks'} for row in rows],indent=2),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as exc:contract.put(OUT,'failure.json',dict(type=type(exc).__name__,message=str(exc)));raise
