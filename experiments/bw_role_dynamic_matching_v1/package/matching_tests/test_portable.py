"""Small explicitly synthetic provenance fixtures; NOT genuine notebook arrays."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
from dynamic_role import portable as p
from helpers import setup
from multiscale_role.short import array_sha


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return path


def engineering_export(tmp_path,monkeypatch):
    from hsgd.bw_preflight import common as c
    from hsgd.kernel import MarginalKernel
    from multiscale_role.legacy import ZIP_SHA
    case,l,pf,k,g,prob,state=setup();kernel=MarginalKernel(l.prior,c.schedule(1000))
    root=tmp_path/'old_export';root.mkdir();short=tmp_path/'old_short';short.mkdir();source=tmp_path/'old_source';source.mkdir()
    package=tmp_path/'engineering_package';reference=package/'data/IMDB_movie_reference.npz';reference.parent.mkdir(parents=True)
    np.savez(reference,fixture=np.array([1]))
    refsha=sha(reference);monkeypatch.setattr(c,'PACKAGE',package);monkeypatch.setattr(c,'config',lambda:dict(imdb_sha256=refsha))
    inputs={};rows=[];oldrecords=[]
    def add(path):inputs[str(path.resolve())]=sha(path)
    for index,(arm,t) in enumerate((a,t) for a in ('A','B') for t in (500,20,1)):
        folder=source/(arm+str(t));folder.mkdir();plan=c.plan_for(case,[2026101701,t,61],20);h=hashlib.sha256();block=np.full(l.families.shape,-1,np.int64)
        for r in range(20):
            ids,f,query=plan.block(r);h.update(ids.tobytes());h.update(f.tobytes());h.update(query.tobytes());block[ids]=r
        collection=write(folder/'collection.json',dict(query_seed=[2026101701,t,61],time=t,next_time=t-1,query_rounds=20,query_sha256=h.hexdigest()))
        report=write(folder/'genre_baseline.json',dict(ENGINEERING=True))
        config=write(folder/'config.json',dict(ENGINEERING=True));src=write(folder/'source_manifest.json',dict(ENGINEERING=True))
        checkpoint=folder/'private_checkpoint_NOT_DELIVERED.pt';cp_sha='c'*64
        data=dict(state=state,p0=prob,q=kernel.reverse(prob,state,l.families,t-1,t),slots=l.slots,families=l.families,
            types=l.types,prior=l.prior,alpha=kernel.alpha,query_block=block,t=np.array(t))
        array=root/(arm+str(t)+'.npz');np.savez(array,**data)
        meta=dict(arm=arm,model_arm=arm,t=t,T=1000,query_rounds=20,sampling_seed=2026101701,completed_updates=6000,final_weight=True,
            reference_sha256=refsha,checkpoint_sha256=cp_sha,config_sha256=sha(config),source_manifest_sha256=sha(src),
            legacy_collection=str(collection),legacy_collection_sha256=sha(collection),legacy_report=str(report),legacy_report_sha256=sha(report),
            legacy_query_sha256=h.hexdigest(),conversion=dict(source_zip_sha256=ZIP_SHA),array_sha256={k:array_sha(v) for k,v in data.items()})
        metadata=write(root/(arm+str(t)+'.json'),meta)
        row=dict(arm=arm,t=t,reference_sha256=refsha,npz=array.name,npz_sha256=sha(array),metadata=metadata.name,metadata_sha256=sha(metadata),
            config=str(config),config_sha256=sha(config),source_manifest=str(src),source_manifest_sha256=sha(src),checkpoint=str(checkpoint),checkpoint_sha256=cp_sha)
        rows.append(row)
        for path in (array,metadata,config,src,collection,report):add(path)
        inputs[str(checkpoint)]=cp_sha
        old=dict(index=index,arm=arm,t=t,ENGINEERING=True);oldrecords.append(old);write(short/('case_%02d.json'%index),old)
    m=write(root/'six_states.json',dict(format='legacy_export_v1_1',base_commit='acbbddb5cdbcbaa2ae129bcd59e434e6a298b130',reference_sha256=refsha,records=rows));add(m)
    artifact=write(source/'independent_bank.json',dict(ENGINEERING=True))
    spec=write(root/'original_evaluation.json',dict(original_artifact_source=str(artifact),original_artifact_sha256=sha(artifact)));add(spec);add(artifact)
    summary=write(short/'summary.json',dict(status='complete',completed_records=6,input_hashes_unchanged=True,input_hash_verification_complete=True,input_hashes=inputs,records=oldrecords))
    write(short/'frozen_objective.json',dict(ENGINEERING=True));write(short/'policy.json',dict(ENGINEERING=True))
    # Only this synthetic fixture substitutes the pinned hashes. Released CLI
    # constants remain the three actual user-supplied immutable SHA values.
    monkeypatch.setattr(p,'PINS',dict(six_states=sha(m),original_evaluation=sha(spec),summary=sha(summary)))
    out=tmp_path/'portable';out.mkdir();result=p.pack(root,short,out,lambda:None)
    bundle=p.Portable(out/'portable.json',sha(out/'portable.json'))
    return case,l,kernel,bundle,rows,result


def test_portable_actual_array_query_and_historical_only_weight_contract(tmp_path,monkeypatch):
    case,l,k,b,rows,result=engineering_export(tmp_path,monkeypatch)
    assert result['status']=='complete' and result['private_weights_included']==0
    for row in rows:
        data,meta,receipt=b.record(row,case,k,l)
        assert receipt['q_max_error']<5e-14 and 'historical receipts ONLY' in receipt['source_authentication']
    assert not list(b.root.rglob('*.pt'))
    b.rehash()


@pytest.mark.parametrize('mutation',['seed','slots','families','source_hash','p0'])
def test_reject_semantic_or_array_tampering(tmp_path,monkeypatch,mutation):
    case,l,k,b,rows,_=engineering_export(tmp_path,monkeypatch);row=rows[-1]
    old_json=b.json
    if mutation in ('seed','source_hash'):
        def changed(path,sha=None):
            value=old_json(path,sha)
            if str(path)==str(b.export/row['metadata']):
                if mutation=='seed':value['sampling_seed']+=1
                else:value['source_manifest_sha256']='d'*64
            return value
        monkeypatch.setattr(b,'json',changed)
    else:
        dest=b.file(b.export/row['npz']);z=dict(np.load(dest))
        if mutation=='slots':z['slots']=z['slots'][:,::-1]
        elif mutation=='families':z['families']=z['families'][::-1]
        else:z['p0']=z['p0']+.01
        np.savez(dest,**z)
        # Force current bytes past the outer-file hash to exercise independent
        # original per-array signatures and semantic checks, not just one gate.
        b.hashed[dest]=sha(dest);b.mapping[str(b.export/row['npz'])]['sha256']=sha(dest);row['npz_sha256']=sha(dest)
    with pytest.raises(ValueError):b.record(row,case,k,l)


def test_portable_payload_mutation_and_receipt_change_refused(tmp_path,monkeypatch):
    case,l,k,b,rows,_=engineering_export(tmp_path,monkeypatch)
    dest=b.file(b.export/rows[0]['metadata']);dest.write_text('{}')
    with pytest.raises(ValueError):b.rehash()
    manifest=json.loads(b.path.read_text());manifest['pins']['six_states']='x'*64;b.path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):p.Portable(b.path,sha(b.path))


def test_protected_directory_and_symlink_no_write(tmp_path):
    old=tmp_path/'old';old.mkdir();link=tmp_path/'link';link.symlink_to(old,target_is_directory=True)
    for out in (old/'new',link/'new',old.parent):
        with pytest.raises((ValueError,FileExistsError)):p.protect(out,[old])
    assert list(old.iterdir())==[]
