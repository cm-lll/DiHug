"""Package existing bytes only; no torch/model import or scientific execution."""
import ast, csv, hashlib, io, json, pathlib, shutil, subprocess, zipfile
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = pathlib.Path('/data2/lyh/gnn_project/DiHug')
GIT = ROOT/'research_branches/typed_structure_192_oabc_formal_v1/review_repo'
WORK = ROOT/'research_branches/small_graph_source_migration'
DEST = WORK/'experiments/small_graph_source_migration_v1'
OUT = ROOT/'results/small_graph_source_migration_20261030'
def sha(b): return hashlib.sha256(b).hexdigest()
def read(p): return json.loads(pathlib.Path(p).read_text())
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def write(p,b):
    p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists() and p.read_bytes()!=b:raise ValueError(('would overwrite different payload',str(p)))
    p.write_bytes(b)
def git_bytes(commit,path):
    return subprocess.check_output(['git','-C',str(GIT),'show',f'{commit}:{path}'])

def main():
    DEST.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    sources=[
      ('ac_screen','472e533fd28cfc067702b1b75082ac5105140a2d','typed_structure_final_screen_v1',ROOT/'results/typed_structure_final_screen_formal_v1_20261029T0325/source_binding.json'),
      ('early_count_stage500','974e5dca8cbdd05ffddc53fd599ae815828d8b56','typed_structure_count_aux_stage500_v1',ROOT/'results/count_aux_stage500_20261029T205144/source_manifest.json'),
      ('early_count_length','c3f9ed007167484e4d336b2768ea2376b92fcb6e','typed_structure_count_aux_length_v1',ROOT/'results/count_aux_length_20261029T231348/source_manifest.json')]
    provenance=[];omitted=[];binding_checks=0
    allowed={'.py','.json','.md','.npz','.txt','.ini','.patch','.yaml','.yml'}
    for name,commit,folder,manifest in sources:
        prefix='experiments/'+folder;binding=read(manifest);files=binding['files']
        allpaths=subprocess.check_output(['git','-C',str(GIT),'ls-tree','-r','--name-only',commit,prefix],text=True).splitlines()
        included=[]
        for path in allpaths:
            rel=path[len(prefix)+1:];p=pathlib.Path(rel)
            # Exclude old result payloads and engineering receipts, retain complete source dependencies.
            keep=(p.suffix in allowed or p.name=='LICENSE') and p.parts[0] not in ('evidence','historical_metadata') and '__pycache__' not in p.parts
            if not keep:omitted.append(dict(snapshot=name,path=rel,reason='non-source/evidence/not needed for runtime'));continue
            b=git_bytes(commit,path)
            if len(b)>2_000_000:raise ValueError(('unexpected large source asset',path,len(b)))
            if rel in files:
                if sha(b)!=files[rel]:raise ValueError(('actual run binding mismatch',name,rel,sha(b),files[rel]))
                binding_checks+=1
            write(DEST/'snapshots'/name/rel,b);included.append(rel)
        missing=sorted(set(files)-set(included))
        # Some NPZ assets were intentionally Git-ignored but bound by their run manifest.
        local_roots={'ac_screen':'typed_structure_final_screen_review','early_count_stage500':'count_aux_stage500_run','early_count_length':'count_aux_length_run'}
        for rel in list(missing):
            p=ROOT/'research_branches'/local_roots[name]/prefix/rel
            if p.is_file() and p.suffix=='.npz' and sha(p.read_bytes())==files[rel]:
                write(DEST/'snapshots'/name/rel,p.read_bytes());included.append(rel);binding_checks+=1;missing.remove(rel)
        # Manifest-bound source must not silently disappear.
        if missing:raise ValueError(('missing bound files',name,missing))
        save(DEST/'provenance'/f'{name}_actual_execution_manifest.json',binding)
        provenance.append(dict(snapshot=name,commit=commit,original_prefix=prefix,actual_execution_manifest_sha256=sha(manifest.read_bytes()),source_file_count=len(included),manifest_bound_file_count=len(files)))

    # IMDB later continued the same science using a small, separately bound adapter.
    commit='7f095e11af2c1441d57012d86d2b7f4ae9bf8058';prefix='experiments/imdb_count_aux_continue_v1'
    for path in subprocess.check_output(['git','-C',str(GIT),'ls-tree','-r','--name-only',commit,prefix],text=True).splitlines():
        p=pathlib.Path(path);rel=path[len(prefix)+1:]
        if p.suffix in allowed and not any(x in pathlib.Path(rel).parts for x in ('evidence','__pycache__')):
            b=git_bytes(commit,path)
            if len(b)<2_000_000:write(DEST/'snapshots/imdb_continue'/rel,b)
    save(DEST/'provenance/imdb_continue_source_version.json',read(ROOT/'results/imdb_count_aux_u6000_20261030T050209/source_version.json'))
    provenance.append(dict(snapshot='imdb_continue',commit=commit,original_prefix=prefix,scientific_parent='early_count_length',note='historical adapter contains expired absolute deadlines and paths'))

    close=ROOT/'results/count_aux_results_closeout_20261030_final'
    ver=read(close/'source_version.json');save(DEST/'provenance/final_reporting_source_version.json',ver)
    for p in (close/'source').iterdir():
        if p.is_file() and p.suffix in ('.py','.md'):
            key='experiments/count_aux_results_closeout_v1/'+p.name
            b=p.read_bytes();assert sha(b)==ver['files'][key];write(DEST/'evaluation/source'/p.name,b)
    for p in (close/'frozen_evaluation_source').rglob('*'):
        if p.is_file() and p.suffix=='.py':write(DEST/'evaluation/frozen_evaluation_source'/p.relative_to(close/'frozen_evaluation_source'),p.read_bytes())
    b=git_bytes(ver['commit'],'experiments/typed_structure_count_aux_length_v1/finalize.py')
    assert sha(b)==ver['files']['experiments/typed_structure_count_aux_length_v1/finalize.py']
    write(DEST/'evaluation/corrected_length_finalize.py',b)
    for name in ('METRIC_DIRECTORY.md','METRICS.md','metric_definition_reference.md','tables/metric_catalog.csv'):
        p=close/name
        if p.exists():write(DEST/'evaluation/definitions'/p.name,p.read_bytes())
    provenance.append(dict(snapshot='final_cpu_reporting',commit=ver['commit'],published_result_commit='b656f75c412b8fa067f800a00eb886cab924486d',note='report-only fixes after scientific execution; not model changes'))

    inv=read(ROOT/'research_branches/typed_structure_final_screen_review/experiments/typed_structure_final_screen_v1/evidence/full_graph_inventory.json')
    assets=[]
    for dataset,o in inv['datasets'].items():
        for key,hkey,target in [('path','sha256',f'{dataset}.npz'),('config_path','config_sha256',f'{dataset}_historical_config.json')]:
            p=pathlib.Path(o[key]);b=p.read_bytes();assert sha(b)==o[hkey]
            write(DEST/'data/full_original'/target,b)
            assets.append(dict(dataset=dataset,file='data/full_original/'+target,sha256=sha(b),bytes=len(b),source=str(p),kind='complete original graph' if key=='path' else 'historical full-graph schema/config; not a new runnable config'))
    save(DEST/'data/full_original/provenance.json',inv)
    save(DEST/'provenance/assets.json',assets)
    save(DEST/'provenance/source_versions.json',provenance)
    save(DEST/'provenance/omitted_files.json',omitted)
    for label,p in [('ac_screen',ROOT/'results/typed_structure_final_screen_formal_v1_20261029T0325/approval.json'),('early_count_length',ROOT/'results/count_aux_length_20261029T231348/approval.json')]:
        # Environment only: never ship a previously active scientific authorization.
        save(DEST/'environment'/f'{label}_recorded_environment.json',read(p)['environment'])

    # Static dependency inventory and path/hardcoding locations, without importing any model.
    issues=[];imports=set();python_count=0
    needles=['/data2/','/home/','/tmp/','18336','18_336','192','2080 Ti','rounds=1','rounds = 1','SOURCE_HASH','DEADLINE','SCIENCE']
    for p in sorted(DEST.rglob('*.py')):
        tree=ast.parse(p.read_text(),filename=str(p));python_count+=1
        for n in ast.walk(tree):
            if isinstance(n,ast.Import):imports.update(x.name.split('.')[0] for x in n.names)
            if isinstance(n,ast.ImportFrom) and n.module:imports.add(n.module.split('.')[0])
        for line,t in enumerate(p.read_text().splitlines(),1):
            hits=[s for s in needles if s in t]
            if hits:issues.append(dict(file=str(p.relative_to(DEST)),line=line,patterns=' | '.join(hits),text=t.strip()[:350]))
    with (DEST/'PORTABILITY_LOCATIONS.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=['file','line','patterns','text']);w.writeheader();w.writerows(issues)
    save(DEST/'environment/static_import_names.json',sorted(imports))
    save(OUT/'build_receipt.json',dict(start_Asia_Shanghai='2026-10-30T12:50:54+08:00',built_Asia_Shanghai=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),bound_source_files_checked=binding_checks,python_syntax_checked=python_count,no_model_loaded=True,no_gpu_used=True,training=0,generation=0,source_bytes_modified=False))
    print(json.dumps(dict(destination=str(DEST),source_bound_checks=binding_checks,python_syntax_checked=python_count,files=len(list(DEST.rglob('*')))),ensure_ascii=False))

if __name__=='__main__':main()
