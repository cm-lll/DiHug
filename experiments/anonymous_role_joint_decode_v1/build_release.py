import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

HERE=Path(__file__).resolve().parent;ROOT=HERE/'package';REPO=HERE.parents[1]
BASE='124d227752fbce651ca20a11e730b19e5de270f7';PREFIX='experiments/bw_role_dynamic_matching_v1_1/package/'
FREEZE='a52f4f4'


def sha(b):return hashlib.sha256(b).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');a=p.parse_args()
    bindings={}
    for name in ('role_degree/role_statistics.py','hsgd/slots.py','data/IMDB_movie_reference.npz','data/IMDB_preprocessing_provenance.json'):
        raw=subprocess.check_output(['git','show',BASE+':'+PREFIX+name],cwd=REPO);bindings[name]=sha(raw)
        if name.startswith('data/') and (ROOT/name).read_bytes()!=raw:raise ValueError('public reference/provenance changed')
    frozen=subprocess.check_output(['git','show',FREEZE+':experiments/anonymous_role_joint_decode_v1/package/frozen_config.json'],cwd=REPO)
    if (ROOT/'frozen_config.json').read_bytes()!=frozen:raise ValueError('pre-result frozen scientific configuration changed')
    if a.prepare:
        env={k:v for k,v in os.environ.items() if not k.startswith('PYTEST_')}
        env.update(PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
        r=subprocess.run([sys.executable,'-m','pytest','--rootdir',str(ROOT),'-c',os.devnull,'tests','-q','--collect-only'],cwd=ROOT,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=60,check=True)
        ids=[s for s in r.stdout.splitlines() if '::' in s and not s.startswith(' ')]
        (ROOT/'test_suite.json').write_text(json.dumps(dict(nodeids=ids),indent=2)+'\n')
    files={str(p.relative_to(ROOT)):p.read_bytes() for p in sorted(ROOT.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts and p.name not in ('manifest.json','SHA256SUMS')}
    if any(n.endswith(('.pt','.pth','.ckpt','.pkl','.pickle','.zip')) for n in files):raise ValueError('private/unplanned binary prohibited')
    manifest=dict(version='anonymous_role_joint_decode_v1',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        definition_baseline=BASE,definition_and_public_data_hashes=bindings,configuration_freeze_commit=subprocess.check_output(['git','rev-parse',FREEZE],cwd=REPO,text=True).strip(),
        independent_runtime_not_copied_from_other_assistant=True,model_probabilities_available=False,
        files={n:dict(sha256=sha(b),bytes=len(b)) for n,b in files.items()})
    raw=(json.dumps(manifest,indent=2)+'\n').encode();(ROOT/'manifest.json').write_bytes(raw);files['manifest.json']=raw
    sums=''.join(sha(b)+'  '+n+'\n' for n,b in sorted(files.items()));(ROOT/'SHA256SUMS').write_text(sums)
    if a.prepare:print(json.dumps(dict(prepared=True,tests=len(json.loads(files['test_suite.json'])['nodeids']))));return
    acceptance=json.loads((ROOT/'evidence/local_acceptance.json').read_text())
    if acceptance.get('status')!='passed' or acceptance.get('cases')!=len(json.loads(files['test_suite.json'])['nodeids']) or acceptance.get('skipped')!=0:raise ValueError('complete real CPU acceptance required')
    files['SHA256SUMS']=sums.encode();out=HERE/'release';out.mkdir(exist_ok=True);archive=out/'dihug_anonymous_role_joint_decode_v1.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for n,b in sorted(files.items()):
            info=zipfile.ZipInfo('dihug_anonymous_role_joint_decode_v1/'+n,date_time=(2026,10,19,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,b)
    h=sha(archive.read_bytes());(out/'SHA256SUMS').write_text(h+'  '+archive.name+'\n')
    print(json.dumps(dict(zip=str(archive),sha256=h,bytes=archive.stat().st_size,source_commit=manifest['source_commit'])))


if __name__=='__main__':main()
