"""Model-free independent audit, including failures/not-run items."""
import resource
from large_common import *
def audit():
    started=time.perf_counter();f=frozen();controller=read(OUT/'controller.json');items=[];paths=[];maxima={};reads=0;status_counts={}
    for meta in f['inputs']:
        ctrl=next((x for x in controller['rows'] if x.get('key')==f"{meta['dataset']}_t{meta['t']}"),dict(status='not_run_supervisor_failure'))
        if time.time()>read(OUT/'activation.json')['deadline_epoch']-540:raise TimeoutError('audit/publication reserve9min')
        name,t=meta['dataset'],meta['t'];folder=OUT/f'{name}_t{t}';case,target,cfg,inv,source=dataset(name)
        assert sha(OUT/meta['path'])==meta['sha256']
        bind=read(OUT/f'{name}_binding.json');assert sha(OUT/f'{name}_binding.json')==f['data_bindings'][name]
        assert bind['data_sha256']==sha(source)==cfg['data']['sha256'] and bind['target_sha256']==target.fingerprint
        with np.load(OUT/meta['path']) as z:state=z['state'];ei=z['edges'];ef=z['families'];types=z['types']
        np.testing.assert_array_equal(types,case['space'].types);ff,loc=case['space'].encode(ei)
        np.testing.assert_array_equal(ff,ef);st=np.zeros(case['space'].total,np.uint8);st[case['space'].offsets[ff]+loc]=1;np.testing.assert_array_equal(st,state)
        degree=np.bincount(ei.ravel(),minlength=len(types));assert int(np.sum(degree*(degree-1)//2))==meta['raw_wedges']
        plan=c.base.c.plan_for(case,[meta['draw_seed'],t,61],20);seen=np.zeros(len(state),np.uint8);hh=hashlib.sha256()
        for b in range(20):
            ids,fam,q=plan.block(b);assert not seen[ids].any();seen[ids]+=1
            for ar in (ids,fam,q):hh.update(np.ascontiguousarray(ar).tobytes())
            encoded,local=case['space'].encode(q);np.testing.assert_array_equal(fam,encoded);np.testing.assert_array_equal(ids,case['space'].offsets[encoded]+local)
        assert hh.hexdigest()==meta['query_sha256'];np.testing.assert_array_equal(seen,case['active'].repeat(case['space'].counts).astype(np.uint8))
        receipt=read(folder/'receipt.json') if (folder/'receipt.json').exists() else None
        observed=read(folder/'paths.json') if (folder/'paths.json').exists() else []
        kernel=c.base.MarginalKernel(case['prior'],c.base.c.schedule(1000));before=None
        for row in observed:
            phase=row['phase'];assert row['completed_full_step'] and row['network_freshly_executed'] and row['blocks']==20
            assert sha(folder/(phase+'_output.npz'))==row['output_sha256'] and sha(folder/(phase+'_readouts.npz'))==row['readouts_sha256']
            with np.load(folder/(phase+'_output.npz')) as z:d={k:z[k].copy() for k in z.files}
            np.testing.assert_array_equal(d['coverage'],seen);p0=d['p0'].astype(float);fam=np.repeat(np.arange(case['n_families']),case['space'].counts)
            q=kernel.reverse(p0,state,fam,t-1,t);np.testing.assert_array_equal(q,d['q'])
            rng=np.random.default_rng([DRAW_SEED,0 if name=='IMDB' else 1,t]);assert rng.bit_generator.state==row['rng_before'];u=rng.random(len(state));assert rng.bit_generator.state==row['rng_after']
            assert hashlib.sha256(u.tobytes()).hexdigest()==row['uniform_sha256'];np.testing.assert_array_equal((u<q).astype(np.uint8),d['state'])
            edges,rel=c.base.c.state_edges(case,d['state']);np.testing.assert_array_equal(edges,d['edges']);np.testing.assert_array_equal(rel,d['families'])
            ff,local=case['space'].encode(edges);ids=case['space'].offsets[ff]+local;assert len(np.unique(ids))==len(ids) and np.all(edges[0]<edges[1])
            from scipy.sparse import coo_matrix
            A=coo_matrix((np.ones(2*len(rel)),(np.r_[edges[0],edges[1]],np.r_[edges[1],edges[0]])),shape=(len(types),len(types))).tocsr()
            assert A.nnz==2*len(rel) and not A.diagonal().any() and (A-A.T).nnz==0
            with np.load(folder/(phase+'_readouts.npz')) as z:alloc={k:z[k].copy() for k in z.files}
            rr=read(folder/(phase+'_reads.json'));assert len(rr)==4
            for typ,r in enumerate(rr):
                L,P,LP,counts=[alloc[f'type{typ}_{k}'] for k in ('L','P','logP','counts')]
                np.testing.assert_array_equal(counts,np.array([n for v,n in target.roles[typ]]));cert=g.verify(P,LP,L,counts,r)
                assert r['context_tail_bound']<=1e-250 and r['context_observed_error']<=r['context_FP32_conversion_bound']+r['context_FP32_dot_bound']+1e-12
                status_counts[r['status']]=status_counts.get(r['status'],0)+1;reads+=1
                for k,v in [('independent_KKT',cert['independent_KKT']),('row',cert['independent_row']),('column',cert['independent_column']),('tail_mass',cert['independent_tail_mass']),('context_tail',r['context_tail_bound'])]:maxima[k]=max(maxima.get(k,0),v or 0)
            if before is not None:
                for k in ('p0','q','coverage','state'):np.testing.assert_array_equal(before[k],d[k])
                assert row['module_timings'].get('new_degree_law_tables',0)==0
            before=d;paths.append(dict(dataset=name,t=t,**row));print('audited',name,t,phase,'all20blocks',flush=True)
        if ctrl['status']=='completed':assert receipt and receipt['status']=='completed' and len(observed)==2
        partial=read(folder/'last_read_receipts.json') if (folder/'last_read_receipts.json').exists() else []
        items.append(dict(dataset=name,t=t,status=ctrl['status'],supervisor_reason=ctrl.get('supervisor_reason'),completed_paths=len(observed),worker_receipt=receipt,planned_query_slots=int(seen.sum()),
            last_attempt_reported_read_count=len(partial),last_attempt_reported_read_statuses=[x.get('status') for x in partial],partial_only_certificates_independently_checked=False if ctrl['status']!='completed' else None))
    save(OUT/'audited_paths.json',paths);save(OUT/'audit.json',dict(status='passed_for_available_evidence',complete_scheduled_panel=all(x['status']=='completed' for x in items),items=items,
        complete_items=sum(x['status']=='completed' for x in items),complete_paths=len(paths),Newton_reads=reads,read_statuses=status_counts,maxima=maxima,
        seconds=time.perf_counter()-started,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,model_calls=0,new_draws_for_science=0,
        checks=['all frozen data/source/input bindings','complete20 block slots/families/query digest','full posterior and same-uniform state replay','schema/simple/symmetric/no-self-loop',
            'every available Newton L/P/logP/capacity/KKT/tail certificate','cold versus fresh repeat exact output and same RNG','failures and missing items retained'],
        limitations=['random weights only','runtime role-value/context error certificates not independently recomputed through model','peak includes full posterior audits/IO','trace duplicates diagnostic posterior','block progress IO not separately timed,inside step residual']))
if __name__=='__main__':audit()
