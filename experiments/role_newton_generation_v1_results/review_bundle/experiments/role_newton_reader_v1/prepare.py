"""Verify sealed small arrays; deterministically reconstruct unsaved large LL, no new states."""
import hashlib,json,os,sys,time
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
HERE=Path(__file__).resolve().parent;EXP=HERE.parent
sys.path.insert(0,str(EXP/'role_logscale_reader_v1'))
import diagnose as old
import numpy as np
OUT=Path('/data2/lyh/gnn_project/DiHug/results/role_newton_20261021_v1')
SOURCE=EXP/'role_logscale_reader_v1_results/evidence'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
    q=p.with_suffix('.tmp')
    q.write_text(json.dumps(x,indent=2,allow_nan=False,default=lambda value:value.item() if isinstance(value,np.generic) else (_ for _ in ()).throw(TypeError(type(value).__name__)))+'\n')
    q.replace(p)
def check():
    if (OUT/'STOP').exists():raise InterruptedError('STOP')
    if time.time()>read(OUT/'activation.json')['deadline_epoch']-120:raise TimeoutError('original60min budget reserve120')
def main():
    check();dest=OUT/'inputs';dest.mkdir();manifest=[]
    expected={n:h for h,n in (line.split('  ',1) for line in (EXP/'role_logscale_reader_v1_results/review_bundle/SHA256SUMS').read_text().splitlines())}
    names=read(old.c.RESULT/'conditions.json')['final_targets'];count=0
    for seed in (2026102011,2026102012,2026102013):
        for name in names:
            for t in (1,8,32,64):
                check();stem=f'seed{seed}/{name}_t{t}';meta=read(SOURCE/(stem+'.json'));passed=meta['status']=='completed'
                suffix='.npz' if passed else '_failure_evidence.npz';p=SOURCE/(stem+suffix)
                assert sha(p)==expected['results/'+stem+suffix]
                rr=meta['allocation'] if passed else read(SOURCE/(stem+'_failure_evidence.json'))['allocations']
                with np.load(p,allow_pickle=False) as z:
                    for typ in range(4):
                        L=z[f'type{typ}_LL'];c=z[f'type{typ}_counts'];out=dest/f'matrix{count:04d}.npz'
                        np.savez_compressed(out,L=L,counts=c)
                        a=rr[typ]
                        manifest.append(dict(index=count,scope='small',seed=seed,condition=name,t=t,type=typ,n=L.shape[0],k=L.shape[1],
                           path=str(out.relative_to(OUT)),sha256=sha(out),source_payload=str(p.relative_to(EXP)),source_sha256=sha(p),
                           source_manifest_expected_sha256=expected['results/'+stem+suffix],fields='original saved arrays',
                           old_scaling=a['scaled'],old_scaling_P_key=f'type{typ}_'+('scaled' if passed else 'unaccepted_scaled')))
                        count+=1
    identities=read(SOURCE/'large_input_identity.json');inventory=read(EXP/'role_target_scalability_v1/evidence/large_graph_inventory.json')
    for name in ('IMDB','PubMed'):
        check();path=EXP/f'joint_cn_dual_rounds_v1_1/package/data/{name}.npz';assert sha(path)==inventory[name]['sha256']
        with np.load(path) as z:case=old.c.base.c.make_case(z['types'],z['edges'].T,tuple(map(tuple,z['schema'])))
        target=old.c.base.extract_target(case);types=case['space'].types;capacity=old.cap.capacities(types,case['space'].pairs)
        receipts=read(SOURCE/f'guard_cost/{name}_reader.json')['rows']
        for t in (20,500):
            check();ei,ef,_=old.c.base.c.global_noisy_edges(case['space'],case['clean'],case['prior'],old.c.base.c.schedule(1000)[t],np.random.default_rng([2026102141,t]))
            source=next(x for x in identities if x['dataset']==name and x['t']==t)
            h=hashlib.sha256(ei.tobytes()+ef.tobytes()).hexdigest();assert h==source['edge_array_sha256']
            d=old.cap.degrees(types,ei,ef,len(case['space'].pairs));tables={}
            for typ,roles in enumerate(target.roles):
                tick=time.perf_counter();ix=types==typ;L=np.zeros((int(ix.sum()),len(roles)));c=np.array([n for v,n in roles])
                for j,(v,_) in enumerate(roles):
                    for r,N in enumerate(capacity[typ]):
                        if not N:continue
                        check();key=(int(N),v[r],float(case['prior'][r]))
                        if key not in tables:tables[key]=old.cap.degree_logpmf(int(N),v[r],old.c.base.c.schedule(1000)[t],float(case['prior'][r]))
                        L[:,j]+=tables[key][d[ix,r]]
                out=dest/f'matrix{count:04d}.npz';np.savez_compressed(out,L=L,counts=c)
                oldr=next(r for r in receipts if r['t']==t and r['type']==typ)
                manifest.append(dict(index=count,scope='large',dataset=name,t=t,type=typ,n=L.shape[0],k=L.shape[1],
                  path=str(out.relative_to(OUT)),sha256=sha(out),fields='deterministic reconstruction; old LL byte hash NOT available',
                  original_state_sha256=h,data_sha256=sha(path),old_state_hash_verified=True,
                  derivation_source_sha256=sha(EXP/'role_logscale_reader_v1/large.py'),degree_law_source_sha256=sha(EXP/'role_specialization_v1/capacity_readout.py'),
                  old_scaling=oldr['scaling'],LL_reconstruction_seconds=time.perf_counter()-tick))
                count+=1
    assert count==544;save(OUT/'input_manifest.json',manifest)
    save(OUT/'preparation_receipt.json',dict(status='passed',small_matrices=528,large_reconstructed=16,new_states=0,
       limitations='no historical large LL hash existed; only source/state hashes independently verified'))
    print('prepared544 matrices;528 archived,16 deterministic reconstructed',flush=True)
if __name__=='__main__':main()
