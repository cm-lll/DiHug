"""Three guard fixes only; all v1.1 mathematical/scientific files protected."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT=Path(__file__).resolve().parent;PACKAGE=ROOT/'package'
FIXED='4740416cfae0235f24a885851e65550cc2832ca6';PREFIX='experiments/bw_role_multiscale_tail_v1_1/package/'
ALLOW={'run_multiscale.py','multiscale_role/legacy.py','INPUT_CONTRACT.md','DEPLOY_README.md','SHA256SUMS','LEGACY_SOURCE_CONTRACT.json'}
EXCLUDE={'SHA256SUMS','guard_manifest.json','v1_1_to_v1_2.diff'}


def sha(data):return hashlib.sha256(data).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');a=p.parse_args()
    repo=Path(subprocess.check_output(['git','rev-parse','--show-toplevel'],cwd=ROOT,text=True).strip())
    paths=subprocess.check_output(['git','ls-tree','-r','-z','--name-only',FIXED,PREFIX],cwd=repo).decode().split('\0')
    base={n[len(PREFIX):]:subprocess.check_output(['git','show',FIXED+':'+n],cwd=ROOT) for n in paths if n}
    if len(base)<100:raise ValueError('fixed v1.1 baseline missing')
    protected={}
    for n,b in base.items():
        if n not in ALLOW:
            if (PACKAGE/n).read_bytes()!=b:raise ValueError('out-of-scope change: '+n)
            protected[n]=sha(b)
    # The adapter change is only redundant verification of the sealed ledger.
    added=b'        from multiscale_role.entry_guard import verify_sealed_ledger\n        verify_sealed_ledger(root,ledger_sha,check)\n'
    if (PACKAGE/'multiscale_role/legacy.py').read_bytes().replace(added,b'')!=base['multiscale_role/legacy.py']:
        raise ValueError('adapter mathematics or other behavior changed')
    suite=json.loads((PACKAGE/'TEST_SUITE.json').read_text())
    old=[n for n in suite['nodeids'] if not n.startswith('tests/test_admission_suite.py::')]
    if len(old)!=112:raise ValueError('original full v1.1 collection not preserved')
    files={}
    for f in sorted(PACKAGE.rglob('*')):
        if not f.is_file() or any(n in f.parts for n in ('__pycache__','.pytest_cache')):continue
        n=str(f.relative_to(PACKAGE))
        if n in EXCLUDE:continue
        if f.suffix in ('.pt','.pth','.ckpt','.pem','.key','.zip'):raise ValueError('private artifact forbidden')
        files[n]=f.read_bytes()
    patch=[]
    for n,b in files.items():
        before=base.get(n,b'')
        if b!=before:
            if b'\0' in b:raise ValueError('changed binary outside scope')
            patch.extend(difflib.unified_diff(before.decode().splitlines(True),b.decode().splitlines(True),fromfile='a/'+n if n in base else '/dev/null',tofile='b/'+n))
    b=''.join(patch).encode();files['v1_1_to_v1_2.diff']=b;(PACKAGE/'v1_1_to_v1_2.diff').write_bytes(b)
    manifest=dict(version='bw_role_multiscale_tail_v1_2',base_review_commit=FIXED,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        protected_v1_1_files=protected,mathematical_objective_policy_and_900_second_limit_unchanged=True,
        frozen_test_total=suite['total'],sealed_ledger_sha256='961e6d93c8194396223ac9062b1658ac87232e82710b1ab3d36e722ed2c00196',
        sealed_ledger_evidence='user supplied sealed-report comparison; real ledger NOT read locally',
        BW_connected=False,real_six_states_executed=False,training=0,scientific_graphs=0,
        files={n:dict(sha256=sha(b),bytes=len(b)) for n,b in sorted(files.items())})
    b=(json.dumps(manifest,indent=2)+'\n').encode();files['guard_manifest.json']=b;(PACKAGE/'guard_manifest.json').write_bytes(b)
    b=''.join(sha(b)+'  '+n+'\n' for n,b in sorted(files.items())).encode();files['SHA256SUMS']=b;(PACKAGE/'SHA256SUMS').write_bytes(b)
    if a.prepare:print(json.dumps(dict(protected=len(protected),total_tests=suite['total'],prepared=True)));return
    evidence=json.loads(files['evidence_guard/acceptance.json'])
    if evidence.get('CPU_cases')!=suite['total'] or evidence.get('status')!='passed_engineering_only' or evidence.get('real_six_states_executed') is not False:
        raise ValueError('guard acceptance missing/inconsistent')
    out=ROOT/'release';out.mkdir(exist_ok=False)
    archive=out/'dihug_sparse_bw_role_multiscale_tail_v1_2.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for n,b in sorted(files.items()):
            info=zipfile.ZipInfo('dihug_sparse_bw_role_multiscale_tail_v1_2/'+n,date_time=(2026,10,18,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,b)
    checks=sha(archive.read_bytes())+'  '+archive.name+'\n'
    (out/(archive.name+'.sha256')).write_text(checks);(out/'SHA256SUMS').write_text(checks)
    print(json.dumps(dict(zip=str(archive),sha256=sha(archive.read_bytes()),bytes=archive.stat().st_size),indent=2))


if __name__=='__main__':main()
