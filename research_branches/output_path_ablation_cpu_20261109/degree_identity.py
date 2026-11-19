"""Exact path/degree bookkeeping; does not evaluate another candidate loss."""
from run import *
started=time.time();jobs=json.loads((RUN/'evaluation_jobs.json').read_text());records=[];coeff=[];cores={}
for ds in ['IMDB','PubMed']:
    with np.load(RUN/f'{ds}_cpu_cost_panel.npz') as z:cores[ds]=RootedScore(z['types'],z['query'],z['families'],z['reference'],10)
def inspect(core,bits,ident):
    inc=RootedIncremental(core,bits);raw=inc.raw;d=raw[:,:core.r];paths=raw[:,core.base_width:].reshape(core.n,core.r,core.r)
    expected=d.T@d-np.diag(d.sum(0));actual=paths.sum(0)
    error=float(np.max(abs(expected-actual)));assert error<1e-10
    max_typed=0.;max_rebuild=0.;max_zero_sum=0.;empty=0
    pairs={}
    q=core.query.numpy();f=core.families.numpy()
    for r in range(core.r):
        ix=np.flatnonzero(f==r)
        if len(ix):
            kinds={tuple(sorted(core.types[q[:,i]])) for i in ix}
            assert len(kinds)==1,'typed degree distribution not sufficient for this schema'
            pairs[r]=next(iter(kinds))
    for t,ids in core.nodes.items():
        ids=ids.numpy();den=d[ids].sum(0);actual_t=paths[ids].sum(0);predicted=np.zeros_like(actual_t)
        for r,pair in pairs.items():
            if t not in pair:continue
            other=pair[1] if pair[0]==t else pair[0];nid=core.nodes[other].numpy()
            predicted[r]=(d[nid,r,None]*d[nid]).sum(0);predicted[r,r]-=den[r]
        max_typed=max(max_typed,float(np.max(abs(predicted-actual_t))))
        mu=np.divide(predicted,den[:,None],out=np.zeros_like(predicted),where=den[:,None]>0)
        residual=paths[ids]-d[ids,:,None]*mu[None]
        rebuilt=residual+d[ids,:,None]*mu[None]
        max_rebuild=max(max_rebuild,float(np.max(abs(rebuilt-paths[ids]))));max_zero_sum=max(max_zero_sum,float(np.max(abs(residual.sum(0)))))
        empty+=int(np.sum(den==0))
        for r in range(core.r):
            for s in range(core.r):coeff.append(dict(**ident,type=int(t),first_relation=r,second_relation=s,degree_exposure=float(den[r]),path_total=float(actual_t[r,s]),degree_determined_coefficient=float(mu[r,s])))
    assert max_typed<1e-10 and max_rebuild<1e-9 and max_zero_sum<1e-8
    records.append(dict(**ident,global_identity_error=error,typed_identity_error=max_typed,reconstruction_error=max_rebuild,residual_sum_error=max_zero_sum,empty_type_relation_groups=empty))
for ds,core in cores.items():inspect(core,core.reference.numpy(),dict(dataset=ds,train_seed=None,arm='reference',sample_seed=None))
for job in jobs:
    core=cores[job['dataset']];ident={k:job[k] for k in ['dataset','train_seed','arm','sample_seed']}
    with np.load(Path(job['folder'])/'graph.npz') as z:
        a=np.zeros((core.n,core.n));a[z['edges'][0],z['edges'][1]]=1;a+=a.T
    q=core.query.numpy();inspect(core,a[q[0],q[1]],ident)
save('degree_path_identities',records);save('degree_determined_coefficients',coeff)
save('identity_summary',dict(graphs=len(records),global_max_error=max(r['global_identity_error'] for r in records),typed_max_error=max(r['typed_identity_error'] for r in records),
    reconstruction_max_error=max(r['reconstruction_error'] for r in records),residual_zero_sum_max_error=max(r['residual_sum_error'] for r in records),
    CPU_seconds=time.time()-started,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
    scope='algebra on64 archived graphs and2 references; no residual loss scored/selected/trained',
    claim='given full per-type joint degree distribution and this unique-endpoint relation schema, degree coefficients are determined; joint (degree,residual) reconstructs original root profiles'))
assert time.time()-started<120 and not torch.cuda.is_initialized()
print((OUT/'identity_summary.json').read_text())
