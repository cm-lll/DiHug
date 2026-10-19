"""Local-only builder. Preserves baseline package; no network or installation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile
HERE=Path(__file__).resolve().parent;ROOT=HERE/'package';REPO=HERE.parents[1]
BASE='2c1a2f2c0c2a4d82402979a03f41be0ad6b02586'


def sha(b):return hashlib.sha256(b).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    if args.prepare:
        env={k:v for k,v in os.environ.items() if not k.startswith('PYTEST_')}
        env.update(PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
        r=subprocess.run([sys.executable,'-m','pytest','--rootdir',str(ROOT),'-c',os.devnull,'-p','no:cacheprovider','tests','-q','--collect-only'],
            cwd=ROOT,env=env,text=True,capture_output=True,timeout=60,check=True)
        ids=[s for s in r.stdout.splitlines() if '::' in s and not s.startswith(' ')]
        (ROOT/'test_suite.json').write_text(json.dumps(dict(nodeids=ids),indent=2)+'\n')
    old=subprocess.check_output(['git','show',BASE+':experiments/anonymous_role_joint_decode_v1/package/tests/test_prototype.py'],cwd=REPO)
    if (ROOT/'tests/test_prototype.py').read_bytes()!=old:raise ValueError('original27 regressions changed')
    files={str(p.relative_to(ROOT)):p.read_bytes() for p in sorted(ROOT.rglob('*')) if p.is_file() and
           '__pycache__' not in p.parts and '.pytest_cache' not in p.parts and p.name not in ('manifest.json','SHA256SUMS')}
    if any(n.endswith(('.pt','.pth','.ckpt','.pkl','.zip')) for n in files):raise ValueError('private/unplanned binary')
    manifest=dict(version='anonymous_role_joint_decode_v1_1_local',base_commit=BASE,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        source_commit_scope='local Git snapshot; GitHub API review commit may have different metadata; identical file hashes bind the release',
        real_frozen_inputs_available=False,real_AB_compatibility_executed=False,model_calls=0,GPU_use=0,
        original27_source_sha256=sha(old),files={n:dict(sha256=sha(b),bytes=len(b)) for n,b in files.items()})
    raw=(json.dumps(manifest,indent=2)+'\n').encode();files['manifest.json']=raw;(ROOT/'manifest.json').write_bytes(raw)
    sums=''.join(sha(b)+'  '+n+'\n' for n,b in sorted(files.items()));(ROOT/'SHA256SUMS').write_text(sums)
    if args.prepare:print(json.dumps(dict(prepared=True,test_count=len(ids))));return
    accept=json.loads((ROOT/'evidence/local_acceptance.json').read_text())
    if accept.get('status')!='passed' or accept.get('cases')!=len(json.loads(files['test_suite.json'])['nodeids']) or accept.get('skipped')!=0:
        raise ValueError('full current CPU acceptance required')
    files['SHA256SUMS']=sums.encode();release=HERE/'release';release.mkdir(exist_ok=True)
    path=release/'dihug_anonymous_role_joint_decode_v1_1_local.zip'
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        for n,b in sorted(files.items()):
            info=zipfile.ZipInfo('dihug_anonymous_role_joint_decode_v1_1_local/'+n,(2026,10,19,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,b)
    h=sha(path.read_bytes());(release/'SHA256SUMS').write_text(h+'  '+path.name+'\n');print(json.dumps(dict(zip=str(path),sha256=h,bytes=path.stat().st_size)))


if __name__=='__main__':main()
