"""Independent additive release, enforcing unchanged fixed-acbbddb source."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT=Path(__file__).resolve().parent;PACKAGE=ROOT/'package'
BASE=ROOT.parent/'bw_stepwise_role_v1/package'
FIXED='acbbddb5cdbcbaa2ae129bcd59e434e6a298b130'


def sha(data):return hashlib.sha256(data).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');a=p.parse_args()
    protected={}
    repo=Path(subprocess.check_output(['git','rev-parse','--show-toplevel'],cwd=ROOT,text=True).strip())
    paths=[p for p in subprocess.check_output(['git','ls-tree','-r','-z','--name-only',FIXED,'experiments/bw_stepwise_role_v1/package'],cwd=repo).decode().split('\0') if p]
    if len(paths)<60:raise ValueError('fixed base file enumeration missing/incomplete')
    for path in paths:
        name=path.split('/package/',1)[1]
        if name=='SHA256SUMS':continue
        original=subprocess.check_output(['git','show',FIXED+':'+path],cwd=ROOT)
        if (PACKAGE/name).read_bytes()!=original:raise ValueError('old implementation changed: '+name)
        protected[name]=sha(original)
    files={}
    for f in sorted(PACKAGE.rglob('*')):
        if not f.is_file() or any(k in f.parts for k in ('__pycache__','.pytest_cache')):continue
        n=str(f.relative_to(PACKAGE))
        if n in ('SHA256SUMS','multiscale_manifest.json','stepwise_v1_to_multiscale_tail_v1.diff'):continue
        if f.suffix in ('.pt','.ckpt','.pth','.pem'):raise ValueError('private artifact forbidden: '+n)
        files[n]=f.read_bytes()
    patch=[]
    for n,data in files.items():
        if n.startswith(('multiscale_role/','tests/test_multiscale')) or n in ('multiscale.sh','run_multiscale.py','MULTISCALE_ROLE_DESIGN.md','INPUT_CONTRACT.md','PREDECLARED_POLICY.json'):
            patch.extend(difflib.unified_diff([],data.decode().splitlines(True),fromfile='/dev/null',tofile='b/'+n))
    name='stepwise_v1_to_multiscale_tail_v1.diff';data=''.join(patch).encode();files[name]=data;(PACKAGE/name).write_bytes(data)
    manifest=dict(version='bw_role_multiscale_tail_v1',base_commit=FIXED,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        protected_unchanged_files=protected,independent_implementation=True,
        real_six_state_short_test='not_executed_missing_inputs_waiting_review',GPU='not_implemented_or_measured',
        no_training_or_scientific_chain=True,
        files={n:dict(sha256=sha(b),bytes=len(b)) for n,b in sorted(files.items())})
    data=(json.dumps(manifest,indent=2)+'\n').encode();files['multiscale_manifest.json']=data;(PACKAGE/'multiscale_manifest.json').write_bytes(data)
    checks=''.join(sha(data)+'  '+n+'\n' for n,data in sorted(files.items())).encode()
    (PACKAGE/'SHA256SUMS').write_bytes(checks);files['SHA256SUMS']=checks
    if a.prepare:print('prepared additive patch and hashes; protected old files',len(protected));return
    accepted=json.loads(files['evidence_multiscale/acceptance.json'])
    if accepted.get('CPU_regression')!='passed' or accepted.get('real_six_states_tested') is not False:
        raise ValueError('missing or misleading scoped acceptance')
    out=ROOT/'release';out.mkdir(exist_ok=False)
    archive=out/'dihug_sparse_bw_role_multiscale_tail_v1.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for n,data in sorted(files.items()):
            info=zipfile.ZipInfo('dihug_sparse_bw_role_multiscale_tail_v1/'+n,date_time=(2026,10,18,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,data)
    (out/'SHA256SUMS').write_text(sha(archive.read_bytes())+'  '+archive.name+'\n')
    print(json.dumps(dict(zip=str(archive),sha256=sha(archive.read_bytes()),bytes=archive.stat().st_size,protected=len(protected)),indent=2))


if __name__=='__main__':main()
