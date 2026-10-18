"""Build independent deployment; verify ALL original package bytes first."""
import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

HERE=Path(__file__).resolve().parent;ROOT=HERE/'package';REPO=HERE.parents[1]
BASE='7049e64b108f2ef434c40815457c74037c4455eb'
PREFIX='experiments/bw_role_multiscale_tail_v1_2/package/'


def sha(b):return hashlib.sha256(b).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');a=p.parse_args()
    names=[s for s in subprocess.check_output(['git','ls-tree','-r','-z','--name-only',BASE,'--',PREFIX],cwd=REPO,text=True).split('\0') if s]
    if len(names)!=121:raise ValueError('fixed baseline must contain all121 original package files')
    protected={}
    for name in names:
        rel=name[len(PREFIX):];expected=subprocess.check_output(['git','show',BASE+':'+name],cwd=REPO)
        if (ROOT/rel).read_bytes()!=expected:raise ValueError('baseline modified: '+rel)
        protected[rel]=sha(expected)
    env={k:v for k,v in os.environ.items() if not k.startswith('PYTEST_')}
    env.update(PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    if a.prepare:
        r=subprocess.run([sys.executable,'-m','pytest','tests','matching_tests','-q','--collect-only'],cwd=ROOT,env=env,
            text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=60,check=True)
        ids=[s for s in r.stdout.splitlines() if '::' in s and not s.startswith(' ')]
        original=json.loads((ROOT/'TEST_SUITE.json').read_text())
        # The original 133 identities are retained in source and re-collected;
        # don't rely on an obsolete minimum test count to admit this release.
        old_ids=[s for s in ids if s.startswith('tests/')]
        if len(old_ids)!=133:raise ValueError('complete baseline test set required')
        (ROOT/'dynamic_test_suite.json').write_text(json.dumps(dict(nodeids=ids,baseline_cases=len(old_ids),new_cases=len(ids)-len(old_ids)),indent=2)+'\n')
    files={str(path.relative_to(ROOT)):path.read_bytes() for path in sorted(ROOT.rglob('*'))
        if path.is_file() and '__pycache__' not in path.parts and '.pytest_cache' not in path.parts
        and path.name not in ('DYNAMIC_SHA256SUMS','dynamic_manifest.json')}
    for n in files:
        if n.endswith(('.pt','.pth','.ckpt','.pkl','.pickle','.zip')):raise ValueError('unsafe/private/unplanned binary in deployment: '+n)
    new={n:b for n,b in files.items() if n not in protected and not n.startswith('evidence_dynamic/')}
    patch=[]
    for n,b in sorted(new.items()):
        if n!='dynamic_test_suite.json':patch.extend(difflib.unified_diff([],b.decode().splitlines(True),fromfile='/dev/null',tofile='b/'+n))
    (HERE/'from_v1_2.diff').write_text(''.join(patch))
    manifest=dict(version='dynamic_anonymous_joint_role_v1',baseline=BASE,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip(),
        protected_unchanged=protected,baseline_files=len(protected),
        frozen_model_loss_noise_posterior_query_unchanged=True,real_six_states_executed=False,BW_executed=False,
        files={n:dict(sha256=sha(b),bytes=len(b)) for n,b in files.items()})
    raw=(json.dumps(manifest,indent=2)+'\n').encode();(ROOT/'dynamic_manifest.json').write_bytes(raw);files['dynamic_manifest.json']=raw
    sums=''.join(sha(b)+'  '+n+'\n' for n,b in sorted(files.items()));(ROOT/'DYNAMIC_SHA256SUMS').write_text(sums)
    if a.prepare:print(json.dumps(dict(protected=len(protected),new_files=len(new),prepared=True)));return
    if not (ROOT/'evidence_dynamic/local_acceptance.json').is_file():raise ValueError('missing actual unpack/regression evidence')
    accepted=json.loads((ROOT/'evidence_dynamic/local_acceptance.json').read_text())
    expected=json.loads((ROOT/'dynamic_test_suite.json').read_text())
    if accepted.get('status')!='passed' or accepted.get('cases')!=len(expected['nodeids']) or accepted.get('skipped')!=0 or accepted.get('exact_collection') is not True:
        raise ValueError('full actual CPU regression not accepted')
    files['DYNAMIC_SHA256SUMS']=sums.encode();release=HERE/'release';release.mkdir(exist_ok=True)
    archive=release/'dihug_bw_dynamic_role_matching_v1.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for n,b in sorted(files.items()):
            info=zipfile.ZipInfo('dihug_bw_dynamic_role_matching_v1/'+n,date_time=(2026,10,19,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,b)
    h=sha(archive.read_bytes());(release/'SHA256SUMS').write_text(h+'  '+archive.name+'\n')
    print(json.dumps(dict(zip=str(archive),sha256=h,bytes=archive.stat().st_size,source_commit=manifest['source_commit'],protected=len(protected))))


if __name__=='__main__':main()
