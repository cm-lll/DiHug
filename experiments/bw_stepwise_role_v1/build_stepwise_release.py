"""Package independent adapter; assert all v1.1 source files are unchanged."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT=Path(__file__).resolve().parent
PACKAGE=ROOT/'package'
BASE=ROOT.parent/'bw_joint_cn_v1_1/package'
BASE_COMMIT='dd53b6431303263e740abedf67e01d7dfc13d15b'


def sha(data):return hashlib.sha256(data).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');a=ap.parse_args()
    original=json.loads(subprocess.check_output(['git','show',BASE_COMMIT+':experiments/bw_joint_cn_v1_1/package/manifest_joint_cn_v1.json'],cwd=ROOT))
    protected={}
    for folder in ('hsgd','role_degree','data','tests'):
        for path in sorted((BASE/folder).rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts and '.pytest_cache' not in path.parts:
                name=str(path.relative_to(BASE));data=path.read_bytes()
                if data!=(PACKAGE/name).read_bytes():raise ValueError('protected original changed: '+name)
                if name in original['files'] and sha(data)!=original['files'][name]['sha256']:
                    raise ValueError('base source differs from fixed manifest: '+name)
                protected[name]=sha(data)
    files={}
    for path in sorted(PACKAGE.rglob('*')):
        if not path.is_file() or any(x in path.parts for x in ('__pycache__','.pytest_cache','.git')):continue
        name=str(path.relative_to(PACKAGE))
        if name in ('SHA256SUMS','stepwise_manifest.json','joint_cn_v1_1_to_stepwise_role_v1.diff'):continue
        if path.suffix in ('.pt','.pth','.ckpt'):raise ValueError('private weights must never enter release')
        files[name]=path.read_bytes()
    patch=[]
    for name,data in files.items():
        if name.startswith(('stepwise_role/','tests/test_stepwise','tests/stepwise_resume')) or name in ('stepwise.sh','STEPWISE_ROLE_README.md'):
            patch.extend(difflib.unified_diff([],data.decode().splitlines(True),fromfile='/dev/null',tofile='b/'+name))
    patch=''.join(patch).encode();files['joint_cn_v1_1_to_stepwise_role_v1.diff']=patch
    (PACKAGE/'joint_cn_v1_1_to_stepwise_role_v1.diff').write_bytes(patch)
    manifest=dict(version='bw_stepwise_role_v1',base_commit=BASE_COMMIT,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        protected_unchanged_files=protected,
        no_training_or_scientific_chain_entry=True,
        BW_weights_and_complete_IMDB_state_not_available=True,
        files={name:dict(sha256=sha(data),bytes=len(data)) for name,data in files.items()})
    data=(json.dumps(manifest,indent=2)+'\n').encode();files['stepwise_manifest.json']=data
    (PACKAGE/'stepwise_manifest.json').write_bytes(data)
    checksum=''.join(sha(data)+'  '+name+'\n' for name,data in sorted(files.items())).encode()
    (PACKAGE/'SHA256SUMS').write_bytes(checksum);files['SHA256SUMS']=checksum
    if a.prepare:print('prepared unchanged-source proofs, patch and hashes');return
    accepted=json.loads(files['evidence_stepwise/acceptance.json'])
    if accepted['CPU_regression']!='passed' or accepted['scientific_quality']!='not_tested':
        raise ValueError('missing or misleading acceptance evidence')
    release=ROOT/'release';release.mkdir(exist_ok=False)
    archive=release/'dihug_sparse_bw_stepwise_role_v1.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for name,data in sorted(files.items()):
            info=zipfile.ZipInfo('dihug_sparse_bw_stepwise_role_v1/'+name,date_time=(2026,10,17,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,data)
    (release/'SHA256SUMS').write_text(sha(archive.read_bytes())+'  '+archive.name+'\n')
    print(json.dumps(dict(zip=str(archive),sha256=sha(archive.read_bytes()),bytes=archive.stat().st_size,
                          protected_files=len(protected)),indent=2))


if __name__=='__main__':main()
