"""Reader-only real schema/role inventory benchmark. No model construction."""
import argparse,resource,sys,time,traceback
from diagnose import c,OUT,EXP,check,CFG,cap
from scaling import scale

def run(name):
    start=time.perf_counter();rows=[];reason=None;status='failed'
    path=EXP/'joint_cn_dual_rounds_v1_1/package/data'/f'{name}.npz'
    inv=c.read(EXP/'role_target_scalability_v1/evidence/large_graph_inventory.json')[name]
    try:
        check();assert c.sha(path)==inv['sha256']
        with c.np.load(path,allow_pickle=False) as z:
            types=z['types'].copy();old=z['edges'].copy();families=z['relations'].copy()-1;pairs=tuple(map(tuple,z['schema']))
        case=c.base.c.make_case(types,old.T,pairs)
        target=c.base.extract_target(case)
        assert len(types)==inv['nodes'] and len(old)==inv['physical_edges']
        for typ,role in enumerate(target.roles):
            assert [(list(v),n) for v,n in role]==[(r['vector'],r['frequency']) for r in inv['types'][typ]['roles']]
        estimates=[dict(type=t,n=len(types[types==t]),k=len(x),matrix_bytes=len(types[types==t])*len(x)*8,
                 estimated_workspace_bytes=len(types[types==t])*len(x)*128) for t,x in enumerate(target.roles)]
        c.save(OUT/f'{name}_resource_estimate.json',dict(types=estimates,source_sha256=inv['sha256'],no_DP_allocated=True,
            convolution_terms_per_time=inv['convolution_terms_per_time'],one_time_cache_bytes=inv['likelihood_cache_bytes_per_time']))
        capacity=cap.capacities(types,pairs);alpha=c.base.c.schedule(1000)
        for t in CFG['large_times']:
            check();tick=time.perf_counter()
            ei,ef,_=c.base.c.global_noisy_edges(case['space'],case['clean'],case['prior'],alpha[t],c.np.random.default_rng([CFG['large_noise_seed'],t]))
            degree=cap.degrees(types,ei,ef,len(pairs));noise_seconds=time.perf_counter()-tick
            tables={}
            # One smallest legal table probe, then the unchanged full exact law.
            tick=time.perf_counter();probe=cap.degree_logpmf(18,4,alpha[t],float(case['prior'][case['active']][0]));probe_seconds=time.perf_counter()-tick
            for typ,roles in enumerate(target.roles):
                check();tick=time.perf_counter();vec=c.np.array([v for v,n in roles]);counts=c.np.array([n for v,n in roles]);ix=types==typ
                ll=c.np.zeros((int(ix.sum()),len(roles)));conv=0.;new=0
                for j,v in enumerate(vec):
                    for r,N in enumerate(capacity[typ]):
                        if not N:continue
                        check();key=(int(N),int(v[r]),float(case['prior'][r]))
                        if key not in tables:
                            ts=time.perf_counter();tables[key]=cap.degree_logpmf(key[0],key[1],alpha[t],key[2]);conv+=time.perf_counter()-ts;new+=1
                        ll[:,j]+=tables[key][degree[ix,r]]
                llseconds=time.perf_counter()-tick
                p,receipt=scale(ll,counts,check);warm=[]
                for repeat in range(2):
                    check();ts=time.perf_counter();pp,rr=scale(ll,counts,check);warm.append(time.perf_counter()-ts)
                    assert rr['status']==receipt['status']
                    if p is not None:c.np.testing.assert_array_equal(pp,p)
                row=dict(dataset=name,type=typ,type_name=inv['types'][typ]['name'],t=t,n=ll.shape[0],k=ll.shape[1],
                    original_forward_noise=True,noise_seconds=noise_seconds,probe_seconds=probe_seconds,
                    loglikelihood_seconds=llseconds,degree_convolution_seconds=conv,lookup_residual_seconds=llseconds-conv,
                    newly_constructed_tables=new,one_time_cache_bytes=sum(v.nbytes for v in tables.values()),
                    scaling=receipt,warm_scaling_seconds=warm,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
                rows.append(row);c.save(OUT/f'{name}_reader_partial.json',dict(rows=rows));print(name,t,typ,receipt['status'],receipt.get('iterations'),llseconds,flush=True)
            del tables
        status='completed'
    except BaseException as exc:reason=repr(exc);traceback.print_exc()
    finally:c.save(OUT/f'{name}_reader.json',dict(status=status,reason=reason,rows=rows,wall_seconds=time.perf_counter()-start,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,model_constructed=False,DP_constructed=False))
    return int(status!='completed')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset',choices=['IMDB','PubMed']);sys.exit(run(p.parse_args().dataset))
