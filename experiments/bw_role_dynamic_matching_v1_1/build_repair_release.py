"""Independent v1.1 repair ZIP; byte-protect math and all nonallowed base files."""
import argparse
import ast
import difflib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

HERE=Path(__file__).resolve().parent;ROOT=HERE/'package';REPO=HERE.parents[1]
BASE='50c6ab1d9df245cd103fbacd42dfeb7298d417c9'
PREFIX='experiments/bw_role_dynamic_matching_v1/package/'
ALLOWED={'run_dynamic.py','dynamic_role/runner.py'}
GENERATED={'DYNAMIC_SHA256SUMS','dynamic_manifest.json','dynamic_test_suite.json'}


def sha(b):return hashlib.sha256(b).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');a=p.parse_args()
    names=[n for n in subprocess.check_output(['git','ls-tree','-r','-z','--name-only',BASE,'--',PREFIX],cwd=REPO,text=True).split('\0') if n]
    if len(names)!=146:raise ValueError('complete v1 baseline must have146 package files')
    originals={};protected={}
    for n in names:
        rel=n[len(PREFIX):];b=subprocess.check_output(['git','show',BASE+':'+n],cwd=REPO);originals[rel]=b
        if rel not in ALLOWED|GENERATED:
            if (ROOT/rel).read_bytes()!=b:raise ValueError('out-of-scope baseline change: '+rel)
            protected[rel]=sha(b)
    old_tree=ast.parse(originals['dynamic_role/runner.py'].decode())
    new_tree=ast.parse((ROOT/'dynamic_role/runner.py').read_text())
    def scientific_parts(tree):
        return {getattr(n,'name','POLICY'):ast.dump(n,include_attributes=False) for n in tree.body
            if isinstance(n,ast.FunctionDef) and n.name in {'distributions','genre_law','compare','resolve_original'}
            or isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='POLICY' for t in n.targets)}
    if scientific_parts(old_tree)!=scientific_parts(new_tree):raise ValueError('scientific policy/evaluation definitions changed')
    env={k:v for k,v in os.environ.items() if not k.startswith('PYTEST_')}
    env.update(PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    if a.prepare:
        collected=subprocess.run([sys.executable,'-m','pytest','tests','matching_tests','-q','--collect-only'],cwd=ROOT,env=env,
            text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=60,check=True)
        ids=[s for s in collected.stdout.splitlines() if '::' in s and not s.startswith(' ')]
        old=json.loads(originals['dynamic_test_suite.json'])['nodeids']
        if [i for i in ids if i in old]!=old:raise ValueError('all original172 test identities must remain in original order')
        (ROOT/'dynamic_test_suite.json').write_text(json.dumps(dict(nodeids=ids,baseline_cases=172,new_cases=len(ids)-172),indent=2)+'\n')
    files={str(p.relative_to(ROOT)):p.read_bytes() for p in sorted(ROOT.rglob('*'))
        if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts and p.name not in {'DYNAMIC_SHA256SUMS','dynamic_manifest.json'}}
    for n in files:
        if n.endswith(('.pt','.pth','.ckpt','.pkl','.pickle','.zip')):raise ValueError('private/unsafe binary not allowed: '+n)
    changes=[]
    for n,b in sorted(files.items()):
        if n in GENERATED or n.startswith('evidence_cli_v1_1/'):continue
        before=originals.get(n,b'')
        if before!=b:
            changes.extend(difflib.unified_diff(before.decode().splitlines(True),b.decode().splitlines(True),
                fromfile='a/'+n if n in originals else '/dev/null',tofile='b/'+n))
    (HERE/'from_dynamic_v1.diff').write_text(''.join(changes))
    manifest=dict(version='dynamic_anonymous_joint_role_v1_1',baseline=BASE,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        allowed_changes=sorted(ALLOWED),protected_unchanged=protected,protected_files=len(protected),
        mathematical_and_scientific_settings_unchanged=True,scientific_policy_and_evaluation_AST_equal=True,
        real_six_states_read=False,BW_executed=False,
        files={n:dict(sha256=sha(b),bytes=len(b)) for n,b in files.items()})
    raw=(json.dumps(manifest,indent=2)+'\n').encode();(ROOT/'dynamic_manifest.json').write_bytes(raw);files['dynamic_manifest.json']=raw
    sums=''.join(sha(b)+'  '+n+'\n' for n,b in sorted(files.items()));(ROOT/'DYNAMIC_SHA256SUMS').write_text(sums)
    if a.prepare:print(json.dumps(dict(prepared=True,protected_files=len(protected),tests=len(json.loads(files['dynamic_test_suite.json'])['nodeids']))));return
    accepted=json.loads((ROOT/'evidence_cli_v1_1/local_acceptance.json').read_text())
    count=len(json.loads(files['dynamic_test_suite.json'])['nodeids'])
    if accepted.get('status')!='passed' or accepted.get('cases')!=count or accepted.get('skipped')!=0 or accepted.get('exact_collection') is not True:
        raise ValueError('complete actual CPU evidence required')
    files['DYNAMIC_SHA256SUMS']=sums.encode();release=HERE/'release';release.mkdir(exist_ok=True)
    path=release/'dihug_bw_dynamic_role_matching_v1_1.zip'
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        for n,b in sorted(files.items()):
            info=zipfile.ZipInfo('dihug_bw_dynamic_role_matching_v1_1/'+n,date_time=(2026,10,19,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,b)
    h=sha(path.read_bytes());(release/'SHA256SUMS').write_text(h+'  '+path.name+'\n')
    print(json.dumps(dict(zip=str(path),sha256=h,bytes=path.stat().st_size,source_commit=manifest['source_commit'],protected_files=len(protected))))


if __name__=='__main__':main()
