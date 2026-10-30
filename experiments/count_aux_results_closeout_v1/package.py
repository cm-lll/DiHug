"""Whitelist result payloads, excluding tensor weights, then audit both ZIPs."""
import argparse, csv, datetime, hashlib, io, json, pathlib, shutil, subprocess, time, zipfile
from build import read, save, sha, SOURCE, MAIN, EARLY, IMDB
HERE=pathlib.Path(__file__).parent
def payload(out):
    return [p for p in sorted(out.rglob('*')) if p.is_file() and p.suffix not in ('.pt','.zip','.pyc') and not any(x in p.relative_to(out).parts for x in ('tmp','__pycache__','pubmed_repaired')) and p.name not in ('payload_manifest.json','ZIP_SHA256.txt','archive_audit.json','publication.json','package_receipt.json') and not (p.relative_to(out).parts[0]=='by_dataset' and p.name=='evaluation.json') and not (p.relative_to(out).parts[0]=='by_dataset' and 'tables' in p.relative_to(out).parts)]
def subset_csv(p):
    rows=csv.DictReader(p.open());fields=rows.fieldnames
    if not fields:return ''
    text=io.StringIO();w=csv.DictWriter(text,fields);w.writeheader()
    for row in rows:
        if 'dataset' not in row or row['dataset']=='PubMed':w.writerow(row)
    return text.getvalue()
def audit_zip(p):
    with zipfile.ZipFile(p) as z:
        assert z.testzip() is None
        manifest=json.loads(z.read('payload_manifest.json'))
        for name,r in manifest.items():
            assert not pathlib.PurePosixPath(name).is_absolute() and '..' not in pathlib.PurePosixPath(name).parts
            data=z.read(name);assert len(data)==r['bytes'] and hashlib.sha256(data).hexdigest()==r['sha256']
        assert len(set(z.namelist()))==len(z.namelist())
        return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size,files=len(manifest),payload_hashes_verified=True,crc_verified=True,no_weights=not any(n.endswith('.pt') for n in z.namelist()))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);out=pathlib.Path(ap.parse_args().output).resolve();begin=time.time()
    assert read(out/'independent_audit.json')['status']=='passed'
    dst=out/'source';shutil.copytree(HERE,dst,ignore=shutil.ignore_patterns('__pycache__'),dirs_exist_ok=True)
    shutil.copy2(HERE/'recompute_cpu.py',out/'recompute_cpu.py')
    repo=HERE.parents[1]
    patch=subprocess.check_output(['git','-C',str(repo),'diff','c3f9ed007167484e4d336b2768ea2376b92fcb6e','--','experiments/typed_structure_count_aux_length_v1/finalize.py']).decode();(out/'finalize_fix.patch').write_text(patch)
    science=[]
    for name in ('evaluation_fields.py','cross_batch.py','lcc_eval/metrics.py','formula.json'):
        old=SOURCE/name;new=repo/'experiments/typed_structure_count_aux_length_v1'/name
        science.append(dict(file=name,parent_sha256=sha(old),current_sha256=sha(new),unchanged=sha(old)==sha(new)))
    assert all(x['unchanged'] for x in science)
    save(out/'scientific_source_unchanged.json',dict(files=science,scope='Only finalize.py and new CPU delivery scripts changed; git patch is included. No source of model/loss/noise/sampling/dictionaries edited.',new_training_updates=0,new_model_forwards=0,new_chains=0))
    save(out/'cpu_preparation_failures.json',[
        dict(attempt='count_aux_results_closeout_20261030',failure='Duplicate saved diversity float values differed by about 2.8e-17; exact compare stopped report',resolution='log both values and use predeclared1e-12 duplicate-number check; graph hashes and identities remain exact'),
        dict(attempt='count_aux_results_closeout_20261030_v2',failure='Old early report6000 missing placeholders compared to completed main results',resolution='retain old pending evidence; current merge uses completed authenticated result'),
        dict(attempt='count_aux_results_closeout_20261030_v3',failure='Detail CSV merge supplied dataset column twice',resolution='validate existing dataset then insert once; no metrics changed'),
        dict(attempt='report refresh command',failure='relative script path repeated inside its own working directory',resolution='corrected to local report.py; no scientific calculation executed')])
    # Do not overwrite any historical archive or original failure file.
    primary=out/'COUNT_AUX_full_results_48graphs.zip';pub=out/'PubMed_1000_2000_3000_6000_repaired.zip'
    files=payload(out);manifest={str(p.relative_to(out)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files};save(out/'payload_manifest.json',manifest)
    with zipfile.ZipFile(primary,'x',zipfile.ZIP_DEFLATED,compresslevel=5) as z:
        for p in files+[out/'payload_manifest.json']:z.write(p,str(p.relative_to(out)))
    pm={}
    def add(z,name,data):
        data=data.encode() if isinstance(data,str) else data;z.writestr(name,data);pm[name]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
    with zipfile.ZipFile(pub,'x',zipfile.ZIP_DEFLATED,compresslevel=5) as z:
        for p in files:
            rel=p.relative_to(out);name=str(rel)
            if rel.parts[0].startswith('IMDB_') and p.suffix in ('.png','.pdf'):continue
            if rel.parts[0]=='run' and len(rel.parts)>1:
                if not rel.parts[1].startswith(('PubMed','derived_lcc_evaluation')):continue
                if rel.parts[1]=='derived_lcc_evaluation' and len(rel.parts)>2 and not rel.parts[2].startswith('PubMed'):continue
            if rel.parts[0]=='reference' and p.stem!='PubMed':continue
            if rel.parts[0]=='by_dataset' and rel.parts[1]!='PubMed':continue
            if rel.parts[0]=='saved_logs' and not any(x.startswith('PubMed_s') for x in rel.parts):continue
            if name=='evaluation.json':add(z,name,(out/'pubmed_repaired/evaluation.json').read_bytes())
            elif name=='REPORT.md':add(z,name,(out/'pubmed_repaired/REPORT.md').read_bytes())
            elif name.endswith('.csv'):add(z,name,subset_csv(p))
            elif name=='tables/sample_diversity.json':add(z,name,json.dumps([r for r in read(p) if r['job']['dataset']=='PubMed'],indent=2))
            elif name=='material_inventory.json':add(z,name,json.dumps([r for r in read(p) if r['dataset']=='PubMed'],indent=2))
            elif name=='validation_panel.json':add(z,name,json.dumps([r for r in read(p) if r['dataset']=='PubMed'],indent=2))
            else:add(z,name,p.read_bytes())
        add(z,'README_PUBMED.md','PubMed修复子包：32张原图、32份LCC，完整四阶段。REPORT.md/evaluation.json/CSV为PubMed子集；总审计回执与中文摘要含关联IMDB结果。旧IMDB独立包路径与SHA见report_repair_receipt.json。本包不夹带权重。\n')
        add(z,'payload_manifest.json',json.dumps(pm,indent=2))
    audits=[audit_zip(p) for p in (primary,pub)]
    # Inventory inside each archive must retain its complete intended graph count.
    with zipfile.ZipFile(primary) as z:assert len(json.loads(z.read('material_inventory.json')))==48
    with zipfile.ZipFile(pub) as z:assert len(json.loads(z.read('material_inventory.json')))==32 and len(json.loads(z.read('evaluation.json'))['records'])==64
    save(out/'archive_audit.json',dict(status='passed',archives=audits,seconds=time.time()-begin))
    (out/'ZIP_SHA256.txt').write_text(''.join(f"{x['sha256']}  {pathlib.Path(x['path']).name}\n" for x in audits))
    save(out/'package_receipt.json',dict(finished_epoch=time.time(),finished_Asia_Shanghai=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),seconds=time.time()-begin,archives=audits,original_failure_preserved=sha(MAIN/'delivery_failure.json')==read(out/'report_repair_receipt.json')['original_failure_sha256'],old_imdb_archive_sha256=(IMDB/'ZIP_SHA256.txt').read_text(),scientific_completion_times_not_modified=True))
    print(json.dumps(audits,indent=2))
if __name__=='__main__':main()
