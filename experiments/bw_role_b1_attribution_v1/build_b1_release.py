"""Additive sidecar; EVERY fixed v1.2 package file stays byte-identical."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT=Path(__file__).resolve().parent;PACKAGE=ROOT/'package';FIXED='7049e64b108f2ef434c40815457c74037c4455eb'
PREFIX='experiments/bw_role_multiscale_tail_v1_2/package/'
EXCLUDE={'B1_SHA256SUMS','b1_manifest.json','v1_2_to_b1_sidecar.diff'}


def sha(b):return hashlib.sha256(b).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');a=p.parse_args()
    repo=Path(subprocess.check_output(['git','rev-parse','--show-toplevel'],cwd=ROOT,text=True).strip())
    paths=subprocess.check_output(['git','ls-tree','-r','-z','--name-only',FIXED,PREFIX],cwd=repo).decode().split('\0');base={}
    for n in paths:
        if n:base[n[len(PREFIX):]]=subprocess.check_output(['git','show',FIXED+':'+n],cwd=repo)
    if len(base)<120:raise ValueError('fixed v1.2 source missing')
    for n,b in base.items():
        if (PACKAGE/n).read_bytes()!=b:raise ValueError('native mathematical/runtime/test file changed: '+n)
    files={}
    for f in sorted(PACKAGE.rglob('*')):
        if not f.is_file() or any(n in f.parts for n in ('__pycache__','.pytest_cache')):continue
        n=str(f.relative_to(PACKAGE))
        if n in EXCLUDE:continue
        if f.suffix in ('.pt','.pth','.ckpt','.pem','.key','.zip'):raise ValueError('private artifact forbidden')
        files[n]=f.read_bytes()
    patch=[]
    for n,b in files.items():
        if n not in base:
            if b'\0' in b:raise ValueError('new binary forbidden in source package')
            patch.extend(difflib.unified_diff([],b.decode().splitlines(True),fromfile='/dev/null',tofile='b/'+n))
    b=''.join(patch).encode();files['v1_2_to_b1_sidecar.diff']=b;(PACKAGE/'v1_2_to_b1_sidecar.diff').write_bytes(b)
    manifest=dict(version='bw_role_b1_attribution_v1',base_commit=FIXED,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        protected_native_files={n:sha(b) for n,b in base.items()},all_original_v1_2_files_byte_identical=True,
        real_B1_tested=False,BW_connected=False,model_calls=0,training=0,scientific_graphs=0,
        replay='old recorded accepted step only; original total infinity denominator; no guide.apply/backtracking search',
        max_invocation_seconds=900,files={n:dict(sha256=sha(b),bytes=len(b)) for n,b in sorted(files.items())})
    b=(json.dumps(manifest,indent=2)+'\n').encode();files['b1_manifest.json']=b;(PACKAGE/'b1_manifest.json').write_bytes(b)
    b=''.join(sha(v)+'  '+n+'\n' for n,v in sorted(files.items())).encode();files['B1_SHA256SUMS']=b;(PACKAGE/'B1_SHA256SUMS').write_bytes(b)
    if a.prepare:print(json.dumps(dict(protected_native=len(base),prepared=True)));return
    accepted=json.loads(files['evidence_b1/acceptance.json'])
    if accepted.get('status')!='passed_small_CPU_engineering_only' or accepted.get('real_B1_tested') is not False or accepted.get('native_CPU_cases')!=133:
        raise ValueError('missing/misleading local acceptance')
    out=ROOT/'release';out.mkdir(exist_ok=False);archive=out/'dihug_sparse_bw_role_b1_attribution_v1.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for n,b in sorted(files.items()):
            info=zipfile.ZipInfo('dihug_sparse_bw_role_b1_attribution_v1/'+n,date_time=(2026,10,18,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,b)
    checks=sha(archive.read_bytes())+'  '+archive.name+'\n'
    (out/'SHA256SUMS').write_text(checks);(out/(archive.name+'.sha256')).write_text(checks)
    print(json.dumps(dict(zip=str(archive),sha256=sha(archive.read_bytes()),bytes=archive.stat().st_size),indent=2))


if __name__=='__main__':main()
