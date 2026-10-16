"""Freeze the two-fix release against published joint CN v1, no model execution."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT=Path(__file__).resolve().parent
BASE_COMMIT='6379e992c107a1d9c2dd124d3e976d2aa2004f2e'
BASE_PREFIX='experiments/bw_joint_cn_v1/package/'
ALLOWED={'hsgd/bw_preflight/common.py','hsgd/fusion_checks.py','hsgd/losses.py'}


def sha(data):return hashlib.sha256(data).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True)
    p.add_argument('--evidence',type=Path,required=True);p.add_argument('--prepare',action='store_true')
    p.add_argument('--output',type=Path);a=p.parse_args();base=a.base.resolve()
    recorded=subprocess.check_output(['git','show',BASE_COMMIT+':'+BASE_PREFIX+'manifest_joint_cn_v1.json'],cwd=ROOT)
    if (base/'manifest_joint_cn_v1.json').read_bytes()!=recorded:raise ValueError('base manifest is not the fixed review commit')
    baseline=json.loads(recorded)
    for n,r in baseline['files'].items():
        if sha((base/n).read_bytes())!=r['sha256']:raise ValueError('fixed base source changed: '+n)
    files={}
    for folder in ['hsgd','tests','data','role_degree']:
        for path in sorted((ROOT/folder).rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix in ['.py','.json','.npz','']:
                files[str(path.relative_to(ROOT))]=path.read_bytes()
    for n in ['run.py','run.sh','build_release.py','pytest.ini','说明.txt','DESIGN_AND_STAGED_PLAN.md',
              'REVIEW_FIXES.md','pyg_inspector_compat.diff','cn_v2_to_joint_v1.diff']:
        files[n]=(ROOT/n).read_bytes()
    protected={}
    for n,data in files.items():
        protected_group=n.startswith(('hsgd/','role_degree/','data/')) or n in [
            'run.py','run.sh','tests/test_contract.py','tests/test_fusion.py',
            'DESIGN_AND_STAGED_PLAN.md','pyg_inspector_compat.diff','cn_v2_to_joint_v1.diff']
        if protected_group and n not in ALLOWED:
            if data!=(base/n).read_bytes():raise ValueError('out-of-scope change: '+n)
            protected[n]=sha(data)
    patch=[]
    for n,data in sorted(files.items()):
        before=(base/n).read_bytes() if (base/n).is_file() else b''
        if before!=data:
            patch.extend(difflib.unified_diff(before.decode().splitlines(True),data.decode().splitlines(True),
                fromfile='a/'+n if before else '/dev/null',tofile='b/'+n))
    patch=''.join(patch).encode()
    (ROOT/'joint_cn_v1_to_v1_1.diff').write_bytes(patch);files['joint_cn_v1_to_v1_1.diff']=patch
    evidence_names=['input_binding.json','acceptance.json','tests_final.xml','tests_final.log','resources_final.txt']
    for n in evidence_names:
        path=a.evidence/n
        if path.is_file():
            data=path.read_bytes();files['evidence/'+n]=data
            (ROOT/'evidence').mkdir(exist_ok=True);(ROOT/'evidence'/n).write_bytes(data)
    if not a.prepare:
        for n in evidence_names:
            if 'evidence/'+n not in files:raise ValueError('missing required evidence: '+n)
        accepted=json.loads(files['evidence/acceptance.json'])
        if accepted.get('status')!='passed_two_fixes_only' or accepted.get('original_cases')!=27 or accepted.get('new_cases')!=12:
            raise ValueError('acceptance not complete')
    manifest=dict(version='bw_joint_cn_v1_1',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        base_review_commit=BASE_COMMIT,allowed_functional_changes=sorted(ALLOWED),
        protected_unchanged_files=protected,architecture_noise_query_sampling_unchanged=True,
        loss='sum exact-support checked stable two-class CE * original R/S',
        no_BW_or_formal_training_or_scientific_sampling=True,
        files={n:dict(sha256=sha(b),bytes=len(b)) for n,b in sorted(files.items())})
    (ROOT/'manifest_joint_cn_v1.json').write_text(json.dumps(manifest,indent=2)+'\n')
    files['manifest_joint_cn_v1.json']=(ROOT/'manifest_joint_cn_v1.json').read_bytes()
    checks=''.join(sha(data)+'  '+n+'\n' for n,data in sorted(files.items()))
    (ROOT/'SHA256SUMS').write_text(checks)
    if a.prepare:print('prepared hashes and relative patch; no model execution');return
    if a.output is None:p.error('--output required')
    a.output.mkdir(exist_ok=False,parents=True);files['SHA256SUMS']=checks.encode()
    archive=a.output/'dihug_sparse_bw_joint_cn_v1_1.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for n,data in sorted(files.items()):z.writestr('dihug_sparse_bw_joint_cn_v1_1/'+n,data)
    (a.output/'SHA256SUMS').write_text(sha(archive.read_bytes())+'  '+archive.name+'\n')
    print(json.dumps(dict(zip=str(archive),sha256=sha(archive.read_bytes()),bytes=archive.stat().st_size,
        source_commit=manifest['source_commit']),indent=2))


if __name__=='__main__':main()
