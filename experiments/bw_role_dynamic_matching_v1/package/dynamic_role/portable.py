"""Read-only portable migration. Never loads private weights or a denoiser.

Payload hashes are verified; unavailable checkpoint hashes remain explicitly
HISTORICAL RECEIPTS, never a claim that model provenance was reauthenticated.
Original manifests/metadata are copied unchanged, with a separate mapping.
"""
import json
import os
import zipfile
from pathlib import Path
import numpy as np
from multiscale_role.runtime_guard import checked_sha,atomic_json
from multiscale_role.short import read_pinned,resolve,array_sha,verify_original_q

PINS=dict(six_states='25bc708337898e84cf22ceffaa8e39ed815adf4fc916b684503c0afa3c35d7dd',
    original_evaluation='198ff9bfcba007a04f36662fc66a9399578b6772b6a5c19be2394544b9b8ffe0',
    summary='aba246f8cf156f16f5271a1160a63ede4475d87dbf13689665e8f6d74ad1b924')
BASE='7049e64b108f2ef434c40815457c74037c4455eb'


def protect(output,inputs):
    """Run before ANY mkdir/write; ancestor, descendant and symlink overlaps."""
    out=Path(output).resolve()
    for inp in inputs:
        p=Path(inp).resolve()
        if out==p or out in p.parents or p in out.parents:
            raise ValueError('output overlaps protected input: '+str(p))
    if out.exists():raise FileExistsError('fresh independent output directory required')
    return out


def pack(export,short_run,output,check):
    export,short=Path(export).resolve(),Path(short_run).resolve();out=Path(output)
    # Outer launcher checks paths before output creation. This function is only
    # called inside the launcher's owned fresh output directory.
    manifest=read_pinned(export/'six_states.json',PINS['six_states'],check)
    spec=read_pinned(export/'original_evaluation.json',PINS['original_evaluation'],check)
    summary=read_pinned(short/'summary.json',PINS['summary'],check)
    if summary.get('status')!='complete' or summary.get('completed_records')!=6 or summary.get('input_hashes_unchanged') is not True or summary.get('input_hash_verification_complete') is not True:
        raise ValueError('sealed original complete six-state receipt required')
    if manifest.get('format')!='legacy_export_v1_1' or manifest.get('base_commit')!='acbbddb5cdbcbaa2ae129bcd59e434e6a298b130':
        raise ValueError('original export format/source differs')
    expected={str(Path(p).resolve()):h for p,h in summary['input_hashes'].items()}
    mapping={};receipts={}
    def add(path,sha=None,require_receipt=True):
        check();path=Path(path).resolve();key=str(path)
        if require_receipt:
            if key not in expected or sha is not None and expected[key]!=sha:raise ValueError('sealed summary has no matching source receipt: '+key)
            sha=expected[key]
        actual=checked_sha(path,check)
        if sha is not None and actual!=sha:raise ValueError('source hash changed: '+key)
        if key not in mapping:
            rel='payload/'+actual+'_'+path.name
            dest=out/rel;dest.parent.mkdir(parents=True,exist_ok=True)
            if path.suffix.lower() in ('.pt','.pth','.ckpt','.pkl','.pickle'):raise ValueError('private/unsafe payload refused')
            # Two unchanged source files may have identical content and name.
            # Content-addressed storage shares the verified payload, never
            # overwrites it, and retains BOTH original path mappings.
            if not dest.exists():
                with path.open('rb') as src,dest.open('xb') as dst:
                    while True:
                        check();block=src.read(1024*1024)
                        if not block:break
                        dst.write(block)
            if checked_sha(dest,check)!=actual:raise ValueError('copy verification failed')
            mapping[key]=dict(path=rel,sha256=actual,authentication='actual bytes checked against sealed receipt' if require_receipt else 'actual bytes; binding described by parent receipt')
        return mapping[key]
    add(export/'six_states.json',PINS['six_states']);add(export/'original_evaluation.json',PINS['original_evaluation'])
    add(short/'summary.json',PINS['summary'],False)
    if sorted((r['arm'],r['t']) for r in manifest['records'])!=sorted((a,t) for a in ('A','B') for t in (500,20,1)):
        raise ValueError('exact six original requests required')
    for row in manifest['records']:
        for key in ('npz','metadata','source_manifest','config'):
            add(resolve(export,row[key]),row[key+'_sha256'])
        meta=json.loads(resolve(export,row['metadata']).read_text())
        for key in ('legacy_collection','legacy_report'):add(meta[key],meta[key+'_sha256'])
        # No private checkpoint copy or read. Keep its *original* expected hash.
        cp=str(resolve(export,row['checkpoint']).resolve())
        if expected.get(cp)!=row['checkpoint_sha256'] or meta.get('checkpoint_sha256')!=row['checkpoint_sha256']:
            raise ValueError('historical checkpoint receipts disagree')
        receipts[cp]=dict(sha256=row['checkpoint_sha256'],status='historical receipt only; actual private weight bytes NOT checked or included')
    origin=resolve(export,spec['original_artifact_source']);add(origin,spec['original_artifact_sha256'])
    # Original case JSON is compared to the pinned summary, not freshly promoted
    # to independent provenance. Include unchanged historical method results.
    for i,record in enumerate(summary['records']):
        path=short/('case_%02d.json'%i)
        if json.loads(path.read_text())!=record:raise ValueError('old per-case result differs from sealed summary')
        add(path,require_receipt=False)
    for name in ('frozen_objective.json','policy.json'):add(short/name,require_receipt=False)
    from hsgd.bw_preflight import common as c
    add(c.PACKAGE/'data/IMDB_movie_reference.npz',c.config()['imdb_sha256'],False)
    # The original source manifests are content receipts. No model weights or
    # original executable package is required to run the offline probability test.
    for path,info in mapping.items():
        if checked_sha(path,check)!=info['sha256']:raise ValueError('input changed during migration')
    result=dict(format='dynamic_role_portable_v1',base_commit=BASE,pins=PINS,
        export_original=str(export),short_original=str(short),mapping=mapping,
        private_weight_receipts=receipts,source_authentication='array/reference/query/posterior checked; model provenance historical receipts only',
        converter_sha256=checked_sha(__file__,check),no_model_calls=True)
    atomic_json(out/'portable.json',result,check)
    # A transfer ZIP, not a deployment ZIP. Only explicitly mapped public/data
    # payloads, never checkpoint paths or filesystem-wide globbing.
    temporary=out/'portable_inputs.zip.tmp';archive=out/'portable_inputs.zip'
    with zipfile.ZipFile(temporary,'x',zipfile.ZIP_DEFLATED) as z:
        paths=[out/'portable.json']+sorted(set(out/v['path'] for v in mapping.values()))
        for path in paths:
            check()
            with path.open('rb') as src,z.open(str(path.relative_to(out)),'w') as dst:
                while True:
                    check();chunk=src.read(1024*1024)
                    if not chunk:break
                    dst.write(chunk)
    check();os.replace(temporary,archive)
    return dict(status='complete',payloads=len(mapping),private_weights_included=0,
        portable_sha256=checked_sha(out/'portable.json',check),transfer_zip_sha256=checked_sha(archive,check),
        transfer_zip_bytes=archive.stat().st_size,source_authentication=result['source_authentication'])


class Portable:
    def __init__(self,path,expected_sha,check=lambda:None):
        self.path=Path(path).resolve();self.root=self.path.parent;self.check=check;self.hashed={}
        self.manifest=read_pinned(self.path,expected_sha,check);self.hashed[self.path]=expected_sha
        if self.manifest.get('format')!='dynamic_role_portable_v1' or self.manifest.get('pins')!=PINS or self.manifest.get('base_commit')!=BASE:
            raise ValueError('portable input contract differs')
        self.mapping=self.manifest['mapping']
        self.export=Path(self.manifest['export_original']);self.short=Path(self.manifest['short_original'])
        self.original=self.json(self.export/'six_states.json',PINS['six_states'])
        self.spec=self.json(self.export/'original_evaluation.json',PINS['original_evaluation'])
        self.old=self.json(self.short/'summary.json',PINS['summary'])
        if self.old.get('status')!='complete' or self.old.get('completed_records')!=6:raise ValueError('old panel incomplete')
        if (self.old.get('input_hashes_unchanged') is not True or self.old.get('input_hash_verification_complete') is not True or
            sorted((r['arm'],r['t']) for r in self.old.get('records',[]))!=sorted((a,t) for a in ('A','B') for t in (500,20,1))):
            raise ValueError('sealed old panel receipt has missing/duplicate cases or incomplete verification')
        for p,entry in self.mapping.items():self.file(p,entry['sha256'])
        if sorted((r['arm'],r['t']) for r in self.original['records'])!=sorted((a,t) for a in ('A','B') for t in (500,20,1)):
            raise ValueError('exact six original records required')
        expected={str(Path(p)):h for p,h in self.old['input_hashes'].items()}
        for name in ('six_states','original_evaluation'):
            if expected.get(str(self.export/(name+'.json')))!=PINS[name]:raise ValueError('original summary/manifest pin mismatch')
        for row in self.original['records']:
            for key in ('npz','metadata','source_manifest','config'):
                orig=resolve(self.export,row[key]);self.file(orig,row[key+'_sha256'])
                if expected.get(str(orig))!=row[key+'_sha256']:raise ValueError('payload vs old sealed summary mismatch')
            cp=str(resolve(self.export,row['checkpoint']))
            if expected.get(cp)!=row['checkpoint_sha256'] or self.manifest['private_weight_receipts'].get(cp,{}).get('sha256')!=row['checkpoint_sha256']:
                raise ValueError('historical-only checkpoint receipt mismatch')

    def file(self,original,expected=None):
        key=str(original)
        if key not in self.mapping:raise ValueError('missing portable payload mapping: '+key)
        item=self.mapping[key];path=(self.root/item['path']).resolve()
        if self.root not in path.parents:raise ValueError('payload escapes portable root')
        if expected is not None and item['sha256']!=expected:raise ValueError('payload expected hash differs')
        if path not in self.hashed:self.hashed[path]=checked_sha(path,self.check)
        if self.hashed[path]!=item['sha256']:raise ValueError('portable payload hash mismatch')
        return path

    def json(self,original,sha=None):return json.loads(self.file(original,sha).read_text())

    def rehash(self):
        for p,h in self.hashed.items():
            if checked_sha(p,self.check)!=h:raise ValueError('portable input mutated during diagnostic')

    def record(self,row,case,kernel,layout):
        from hsgd.bw_preflight.common import schedule
        from multiscale_role.legacy import original_query,ZIP_SHA
        meta=self.json(resolve(self.export,row['metadata']),row['metadata_sha256'])
        for k,v in dict(arm=row['arm'],model_arm=row['arm'],t=row['t'],T=1000,query_rounds=20,
                        sampling_seed=2026101701,completed_updates=6000,final_weight=True,
                        reference_sha256=row['reference_sha256']).items():
            if meta.get(k)!=v:raise ValueError('metadata binding differs: '+k)
        for key in ('checkpoint','config','source_manifest'):
            if meta.get(key+'_sha256')!=row[key+'_sha256']:raise ValueError('producer source/config binding differs')
        with np.load(self.file(resolve(self.export,row['npz']),row['npz_sha256']),allow_pickle=False) as z:
            required=('state','p0','q','slots','families','types','prior','alpha','query_block','t')
            if not set(required).issubset(z.files):raise ValueError('missing required offline array')
            data={k:z[k].copy() for k in required}
        for k,v in data.items():
            if meta['array_sha256'].get(k)!=array_sha(v):raise ValueError('original array signature mismatch: '+k)
        for k,v in dict(slots=layout.slots,families=layout.families,types=layout.types,prior=layout.prior,alpha=schedule(1000)).items():
            if not np.array_equal(data[k],v):raise ValueError('canonical layout/schedule differs: '+k)
        from hsgd.bw_preflight.common import config
        if int(data['t'])!=row['t'] or row['reference_sha256']!=config()['imdb_sha256']:
            raise ValueError('time/reference differs')
        col=self.json(meta['legacy_collection'],meta['legacy_collection_sha256'])
        block,coverage,qhash=original_query(case,row['t'],col,self.check)
        if not np.array_equal(block,data['query_block']) or meta['legacy_query_sha256']!=qhash or meta['conversion']['source_zip_sha256']!=ZIP_SHA:
            raise ValueError('full ids/families/query byte digest or provenance differs')
        p,q,error=verify_original_q(data,kernel,layout,row['t'])
        return data,meta,dict(q_max_error=error,full_query_sha256=qhash,
            source_authentication='actual arrays, reference, metadata, query, posterior verified; private weights historical receipts ONLY')
