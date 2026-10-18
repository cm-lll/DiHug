"""Reproducible independent repair release against fixed d1e1d9a; no model."""
import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT=Path(__file__).resolve().parent;PACKAGE=ROOT/'package'
FIXED='d1e1d9a48f4b275c158e9d5d4048a833e2e28f89'
PREFIX='experiments/bw_role_multiscale_tail_v1/package/'
ALLOW={'multiscale_role/core.py','multiscale_role/evaluation.py','multiscale_role/short.py',
       'run_multiscale.py','multiscale.sh','INPUT_CONTRACT.md','SHA256SUMS'}
EXCLUDE={'SHA256SUMS','repair_manifest.json','v1_to_v1_1_repair.diff'}


def sha(b):return hashlib.sha256(b).hexdigest()


def semantic_node(data,kind,name):
    tree=ast.parse(data.decode())
    if kind=='function':return ast.dump(next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)
    return ast.dump(next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)),include_attributes=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');a=p.parse_args()
    repo=Path(subprocess.check_output(['git','rev-parse','--show-toplevel'],cwd=ROOT,text=True).strip())
    paths=subprocess.check_output(['git','ls-tree','-r','-z','--name-only',FIXED,PREFIX],cwd=repo).decode().split('\0')
    base={n[len(PREFIX):]:subprocess.check_output(['git','show',FIXED+':'+n],cwd=repo) for n in paths if n}
    if len(base)<80:raise ValueError('fixed base missing')
    protected={}
    for n,b in base.items():
        if n not in ALLOW:
            if (PACKAGE/n).read_bytes()!=b:raise ValueError('protected original file changed: '+n)
            protected[n]=sha(b)
    for kind,name in [('assign','DESIGN'),('function','freeze'),('function','statistic'),('function','posterior_gradient'),('function','apply')]:
        if semantic_node(base['multiscale_role/core.py'],kind,name)!=semantic_node((PACKAGE/'multiscale_role/core.py').read_bytes(),kind,name):
            raise ValueError('frozen mathematical/controller definition changed: '+name)
    if semantic_node(base['multiscale_role/short.py'],'assign','POLICY')!=semantic_node((PACKAGE/'multiscale_role/short.py').read_bytes(),'assign','POLICY'):
        raise ValueError('scientific decision policy changed')
    files={}
    for f in sorted(PACKAGE.rglob('*')):
        if not f.is_file() or any(n in f.parts for n in ('__pycache__','.pytest_cache')):continue
        n=str(f.relative_to(PACKAGE))
        if n in EXCLUDE:continue
        if f.suffix in ('.pt','.pth','.ckpt','.pem','.key','.zip'):raise ValueError('private artifact forbidden: '+n)
        files[n]=f.read_bytes()
    patch=[]
    for n,b in files.items():
        before=base.get(n,b'')
        if before!=b:
            if b'\0' in b:raise ValueError('unexpected changed binary: '+n)
            patch.extend(difflib.unified_diff(before.decode().splitlines(True),b.decode().splitlines(True),fromfile='a/'+n if n in base else '/dev/null',tofile='b/'+n))
    patch=''.join(patch).encode();files['v1_to_v1_1_repair.diff']=patch;(PACKAGE/'v1_to_v1_1_repair.diff').write_bytes(patch)
    manifest=dict(version='bw_role_multiscale_tail_v1_1',base_review_commit=FIXED,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        protected_original_files=protected,original_test_cases=85,frozen_mathematical_methods_and_policy_AST_unchanged=True,
        scope='interfaces, process protection, exact evaluation reuse and immutable target cache only',
        real_six_states_executed=False,BW_connected=False,model_calls=0,training_updates=0,scientific_graphs=0,
        files={n:dict(sha256=sha(b),bytes=len(b)) for n,b in sorted(files.items())})
    b=(json.dumps(manifest,indent=2)+'\n').encode();files['repair_manifest.json']=b;(PACKAGE/'repair_manifest.json').write_bytes(b)
    checks=''.join(sha(b)+'  '+n+'\n' for n,b in sorted(files.items())).encode();(PACKAGE/'SHA256SUMS').write_bytes(checks);files['SHA256SUMS']=checks
    if a.prepare:print(json.dumps(dict(prepared=True,protected=len(protected),files=len(files))));return
    evidence=json.loads(files['evidence_repairs/acceptance.json'])
    if evidence.get('CPU_regression')!='passed' or evidence.get('real_six_states_tested') is not False or evidence.get('original_cases')!=85:
        raise ValueError('missing or misleading repair acceptance')
    release=ROOT/'release';release.mkdir(exist_ok=False)
    archive=release/'dihug_sparse_bw_role_multiscale_tail_v1_1.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for n,b in sorted(files.items()):
            info=zipfile.ZipInfo('dihug_sparse_bw_role_multiscale_tail_v1_1/'+n,date_time=(2026,10,18,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,b)
    (release/'SHA256SUMS').write_text(sha(archive.read_bytes())+'  '+archive.name+'\n')
    (release/(archive.name+'.sha256')).write_text(sha(archive.read_bytes())+'  '+archive.name+'\n')
    print(json.dumps(dict(zip=str(archive),sha256=sha(archive.read_bytes()),bytes=archive.stat().st_size,protected=len(protected)),indent=2))


if __name__=='__main__':main()
