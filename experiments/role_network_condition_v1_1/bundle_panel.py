"""Publishable synthetic-panel evidence and per-seed fresh toy weights only."""
import argparse,hashlib,json,zipfile
from pathlib import Path

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()

def archive(path,entries,compress=True):
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED,compresslevel=3 if compress else None) as z:
        for source,name in entries:
            info=zipfile.ZipInfo(name,(2026,10,20,0,0,0));info.external_attr=0o100644<<16
            info.compress_type=zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
            with z.open(info,'w') as target,source.open('rb') as f:
                for chunk in iter(lambda:f.read(1048576),b''):target.write(chunk)

def main(run,preflight,dest):
    run=Path(run).resolve();preflight=Path(preflight).resolve();dest=Path(dest).resolve();dest.mkdir(exist_ok=False)
    cfg=json.loads((run/'resolved_config.json').read_text());panel=json.loads((run/'panel.json').read_text())
    assert panel['nodes']==34 and panel['type_counts']==[4,6,18,6] and len(panel['graphs'])==12
    assert cfg['train_seeds']==[2026102011,2026102012,2026102013] and cfg['updates']==1000
    # These are newly trained SYNTHETIC small-graph weights; no private reference/model files are loaded.
    entries=[];hashes={};models=[]
    for p in sorted(run.rglob('*')):
        if p.is_file():
            name='run/'+str(p.relative_to(run));hashes[name]=digest(p)
            if p.suffix!='.pt':entries.append((p,name))
    for p in sorted(preflight.rglob('*')):
        if p.is_file():
            name='preflight/'+str(p.relative_to(preflight));hashes[name]=digest(p);entries.append((p,name))
    for suffix in ['.budget.json','.console.log']:
        p=run.with_name(run.name+suffix);hashes[p.name]=digest(p);entries.append((p,p.name))
    (dest/'input_output_hashes.json').write_text(json.dumps(hashes,sort_keys=True,indent=2))
    entries.append((dest/'input_output_hashes.json','input_output_hashes.json'))
    small=dest/'role_condition_panel_results.zip';archive(small,entries)
    archives=[small]
    for seed in cfg['train_seeds']:
        directory=run/('seed'+str(seed));files=sorted(p for p in directory.rglob('*') if p.is_file()) if directory.exists() else []
        if not files:continue
        # Store without expensive recompression; each seed package remains below GitHub100MB.
        path=dest/('synthetic_seed_'+str(seed)+'.zip');archive(path,[(p,str(p.relative_to(run))) for p in files],compress=False)
        assert path.stat().st_size<100000000,'per-seed archive exceeds public GitHub file limit; do not split/change scientific results'
        archives.append(path)
        models.extend(str(p.relative_to(run)) for p in files if p.suffix=='.pt')
    manifest=dict(run_summary=json.loads((run/'summary.json').read_text()),controller=json.loads((run/'controller.json').read_text()),
        model_scope='fresh synthetic34-node training only; no private/big-graph/notebook weights',
        weights=models,archives={p.name:dict(sha256=digest(p),bytes=p.stat().st_size) for p in archives},
        no_new_training_or_sampling=True,all_planned_training_seeds=cfg['train_seeds'])
    (dest/'release.json').write_text(json.dumps(manifest,indent=2))
    (dest/'SHA256SUMS').write_text(''.join(digest(p)+'  '+p.name+'\n' for p in archives+[dest/'input_output_hashes.json',dest/'release.json']))
    print(json.dumps(manifest,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--preflight',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();main(args.run,args.preflight,args.output)
