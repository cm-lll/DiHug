"""Read sealed validation/chain records. Never import a model or replay a chain.

Only torch.load(map_location='cpu') on state-only payloads is permitted. The
portable evidence contains copied uint8 states, not new graph samples.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,csv,datetime,hashlib,itertools,json,pathlib,shutil,time
import numpy as np

COMMITS=['b5b697361e389cd77d2fb09a84e77c0b6915407d','143e6bf80a9858ec21e512da2f1a03e02e3a5a38','93b41d954fcc46fd3732eb5c2780c76db0568f29']
RUNS=['typed_structure_192_oabc_formal_v1_20261028T090624','typed_structure_192_oabc_seed2_v1_20261028T110507','typed_structure_192_de_formal_v1_20261028T140346']
SEEDS={'IMDB':[2026102811,2026102834],'PubMed':[2026102812,2026102833]}
TIMES=[999,899,799,699,599,499,399,299,199,99,0]
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def read(p):return json.loads(p.read_text())
def csvwrite(p,rows):
    with p.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));writer.writeheader();writer.writerows(rows)
def slot_arrays(types,schema):
    """Independent family-major decoding, following frozen TypedSlots order."""
    edges=[];families=[]
    for f,(a,b) in enumerate(schema):
        ns=np.flatnonzero(types==a);ms=np.flatnonzero(types==b)
        pairs=itertools.combinations(ns,2) if a==b else itertools.product(ns,ms)
        es=[(min(int(u),int(v)),max(int(u),int(v))) for u,v in pairs]
        edges.extend(es);families.extend([f]*len(es))
    return np.asarray(edges,np.int64).T,np.asarray(families,np.int64)
def counts(state,families,nf):
    assert state.shape==families.shape and np.isin(state,[0,1]).all()
    return np.bincount(families[state.astype(bool)],minlength=nf).astype(int)
def read_state(path):
    import torch
    # These are locally sealed state receipts, not arbitrary untrusted pickles.
    p=torch.load(path,map_location='cpu')
    assert 'model' not in p and 'optimizer' not in p
    return p
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',required=True);ap.add_argument('--output',required=True);args=ap.parse_args()
    start=time.time();root=pathlib.Path(args.project).resolve();out=pathlib.Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    source=root/'research_branches/typed_structure_192_de_runtime_review/experiments/typed_structure_192_de_runtime_v1/package'
    cfg=read(source/'config.json');provenance=[];schemas={};totals=[];familyrows=[];trajectories=[];trajectoryfamilies=[];inventory=[];input_rows=[];checks=[];cellpaths={}
    def copy(p,dst):
        dst=out/dst;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
        provenance.append(dict(original_path=str(p),copied_path=str(dst.relative_to(out)),sha256=sha(p),bytes=p.stat().st_size))
    for rel in ['worker.py','normal2.py','probe.py','hsgd/slots.py','hsgd/bw_preflight/common.py','hsgd/losses.py','config.json']:
        copy(source/rel,'frozen_source/'+rel)
    for ds,d in cfg['datasets'].items():
        asset=source/d['file'];assert sha(asset)==d['sha256'];copy(asset,'references/'+ds+'.npz')
        z=np.load(asset,allow_pickle=False);types=z['types'];schema=z['schema'];ei=z['edges'];all_edges,fams=slot_arrays(types,schema)
        pair_to_f={tuple(p):i for i,p in enumerate(schema.tolist())}
        f=np.asarray([pair_to_f[tuple(sorted((int(types[u]),int(types[v]))))] for u,v in ei])
        assert np.array_equal(f,z['relations'].astype(np.int64)-1)  # source reference is 1-based
        refs=np.bincount(f,minlength=len(schema)); assert len(fams)==18336
        schemas[ds]=dict(types=types,schema=schema,edges=all_edges,families=fams,reference=refs,names=[d['type_names'][a]+'--'+d['type_names'][b] for a,b in schema])
        save(out/'references'/f'{ds}_schema.json',dict(pairs=schema.tolist(),family_names=schemas[ds]['names'],reference_edges=refs.tolist(),slot_counts=np.bincount(fams).tolist(),S=18336,active_slots=int(sum(np.bincount(fams)[refs>0]))))
    for runidx,runname in enumerate(RUNS):
        run=root/'results'/runname/'run'
        for vp in sorted(run.glob('*/validation.json')):
            folder=vp.parent;b=read(folder/'binding.json');ds=b.get('dataset_name') or folder.name.split('_')[0];arm=b['arm'];seed=b.get('train_seed',b['config']['seed']);cell=f'{ds}_s{seed}_{arm}';cellpaths[cell]=str(folder)
            assert seed in SEEDS[ds];sc=schemas[ds];ref=sc['reference'];assert b['dataset']['sha256']==cfg['datasets'][ds]['sha256']
            base={'dataset':ds,'train_seed':seed,'arm':arm,'cell':cell,'source_commit':COMMITS[runidx]}
            copy(vp,f'evidence/{cell}/validation.json');copy(folder/'binding.json',f'evidence/{cell}/binding.json')
            v=read(vp);assert [r['updates'] for r in v]==[0,500,1000]
            for i,snap in enumerate(v):
                assert [r['t'] for r in snap['rows']]==[1,500,1000]
                for j,row in enumerate(snap['rows']):
                    ident=dict(base,updates=snap['updates'],noise_t=row['t'],input_sha256=row.get('input_sha256'),source_file=str(vp),json_pointer=f'/{i}/rows/{j}',input_kind='ordinary_forward_noise_from_reference')
                    fam=row['family'];assert len(fam)==10
                    positive=sum(r['positive_count'] for r in fam);negative=sum(r['negative_count'] for r in fam)
                    ps=sum(r['positive_count']*(r['positive_ce'] or 0) for r in fam);ns=sum(r['negative_count']*(r['negative_ce'] or 0) for r in fam)
                    expectation=sum(r['expected_edges'] for r in fam);truth=sum(r['true_edges'] for r in fam)
                    assert truth==int(ref.sum());reconstructed=(ps+ns)/18336
                    assert abs(reconstructed-row['ce_sum_over_S'])<2e-6
                    checks.append(dict(kind='CE_from_class_means',cell=cell,updates=snap['updates'],t=row['t'],absolute_error=abs(reconstructed-row['ce_sum_over_S'])))
                    totals.append(dict(ident,total_CE=row['ce_sum_over_S'],normalization='sumCE / S; S=18336; one validation input, no factor 1/2',S=18336,p0_sum=expectation,reference_edges=truth,p0_signed_bias=expectation-truth,positive_count=positive,negative_count=negative,positive_CE_mean=ps/positive if positive else None,negative_CE_mean=ns/negative if negative else None,q_sum=None,validation_Xt_edge_count=None,missing_fields='q;per-slot p0/logits;validation Xt array/count;separate Xt/query hashes'))
                    input_rows.append(dict(ident))
                    for f,r in enumerate(fam):
                        assert r['family']==f and r['true_edges']==ref[f]
                        familyrows.append(dict(ident,family=f,relation=sc['names'][f],p0_sum=r['expected_edges'],reference_edges=r['true_edges'],p0_signed_bias=r['expected_edges']-r['true_edges'],positive_count=r['positive_count'],negative_count=r['negative_count'],positive_CE_mean=r['positive_ce'],negative_CE_mean=r['negative_ce'],null_reason='no queried samples; fixed zero support' if r['positive_count']+r['negative_count']==0 else None,q_sum=None))
            for sample_seed in [2026102821,2026102822]:
                sf=folder/f'sample_{sample_seed}';log=sf/'steps.jsonl';copy(log,f'evidence/{cell}/sample_{sample_seed}/steps.jsonl');copy(sf/'graph.npz',f'evidence/{cell}/sample_{sample_seed}/graph.npz');copy(sf/'metrics.json',f'evidence/{cell}/sample_{sample_seed}/metrics.json')
                lines=[json.loads(s) for s in log.read_text().splitlines()];assert len(lines)==1000
                assert [r['time'] for r in lines]==list(range(1000,0,-1));assert [r['remaining_time'] for r in lines]==list(range(999,-1,-1))
                keys=sorted(set().union(*(r.keys() for r in lines)));assert all(r['exact_coverage'] for r in lines)
                assert not any(k in keys for k in ['edges','family_edges','p0','q','expected_edges'])
                snapshots=[];states=[];state_sources=[]
                for p in sorted(sf.glob('complete_t*.pt'),reverse=True):
                    # Read state only, verify its own existing receipt (not all weights).
                    receipt=p.with_suffix('.sha256.json');h=sha(p);assert h==read(receipt)['sha256'];s=read_state(p)
                    t=int(s['next_t']);assert int(p.stem[-4:])==t and s['sample_seed']==sample_seed
                    assert s['binding']==b
                    state=np.asarray(s['state']);cs=counts(state,sc['families'],10);assert not cs[ref==0].any()
                    snapshots.append(t);states.append(state);state_sources.append(dict(original_path=str(p),sha256=h,remaining_t=t,source_kind='saved_committed_generated_state'))
                    common=dict(base,sample_seed=sample_seed,state_t=t,reverse_steps_completed=1000-t,produced_by_reverse_t=t+1,state_kind='committed X(t-1), not input Xt of the producing step',source_file=str(p),state_file_sha256=h,state_array_sha256=hashlib.sha256(state.tobytes()).hexdigest())
                    trajectories.append(dict(common,actual_edges=int(cs.sum()),reference_edges=int(ref.sum()),actual_signed_bias=int(cs.sum()-ref.sum()),p0_sum=None,q_sum=None))
                    for f,c in enumerate(cs):trajectoryfamilies.append(dict(common,family=f,relation=sc['names'][f],actual_edges=int(c),reference_edges=int(ref[f]),actual_signed_bias=int(c-ref[f]),p0_sum=None,q_sum=None))
                assert snapshots==TIMES,(cell,snapshots)
                gz=np.load(sf/'graph.npz',allow_pickle=False);expected=sc['edges'][:,states[-1].astype(bool)]
                def eset(e):return set(map(tuple,e.T.tolist()))
                assert eset(expected)==eset(gz['edges']);assert np.array_equal(sc['types'],gz['types'])
                dest=out/f'evidence/{cell}/sample_{sample_seed}/saved_states.npz';np.savez_compressed(dest,state_t=np.asarray(snapshots),state=np.stack(states))
                save(dest.with_suffix('.sources.json'),state_sources)
                inventory.append(dict(base,sample_seed=sample_seed,log_fields=keys,log_rows=1000,states=snapshots,states_count=len(states),missing='initial X1000; intermediate unsaved states; per-slot/logged p0 and q; predicted births/deaths',total_step_seconds=sum(r['step_seconds'] for r in lines)))
    assert len(totals)==216 and len(familyrows)==2160 and len(trajectories)==528 and len(inventory)==48
    grouped={}
    for r in input_rows:grouped.setdefault((r['dataset'],r['train_seed'],r['noise_t']),[]).append(r)
    consistency=[]
    for (ds,seed,t),rows in sorted(grouped.items()):
        hs=sorted(set(r['input_sha256'] for r in rows));assert len(hs)==1
        consistency.append(dict(dataset=ds,train_seed=seed,t=t,arms=sorted(set(r['arm'] for r in rows)),checkpoints=[0,500,1000],rows=len(rows),same_combined_input_hash=True,input_sha256=hs[0]))
    for ds in SEEDS:
        for t in [1,500,1000]:assert len({x['input_sha256'] for x in consistency if x['dataset']==ds and x['t']==t})==2
    csvwrite(out/'validation_total.csv',totals);csvwrite(out/'validation_relations.csv',familyrows);csvwrite(out/'trajectory_total.csv',trajectories);csvwrite(out/'trajectory_relations.csv',trajectoryfamilies)
    save(out/'input_consistency.json',consistency);save(out/'field_inventory.json',inventory);save(out/'source_files.json',provenance);save(out/'cell_sources.json',cellpaths)
    save(out/'analysis_receipt.json',dict(status='complete',started=datetime.datetime.fromtimestamp(start).astimezone().isoformat(),finished=datetime.datetime.now().astimezone().isoformat(),seconds=time.time()-start,validation_rows=len(totals),validation_family_rows=len(familyrows),chains=48,saved_states=len(trajectories),state_family_rows=len(trajectoryfamilies),CE_reconstruction_max_abs_error=max(c['absolute_error'] for c in checks),checks=checks,model_loads=0,forwards=0,optimizer_steps=0,new_graphs=0,gpu_calls=0,software=dict(numpy=np.__version__),source_commits=COMMITS))
    print(json.dumps(dict(output=str(out),seconds=time.time()-start,validation_rows=len(totals),states=len(trajectories))))
if __name__=='__main__':main()
