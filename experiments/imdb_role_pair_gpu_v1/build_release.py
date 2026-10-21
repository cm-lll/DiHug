"""Immutable standard notebook ZIP with evidence and separate minimal patches."""
import hashlib,json,zipfile
from pathlib import Path
R=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    files=[]
    for p in R.rglob('*'):
        if not p.is_file() or '__pycache__' in p.parts or '.pytest_cache' in p.parts:continue
        if p.suffix in ('.zip','.pt','.pth','.pyc','.pem') or p.name in ('release_receipt.json','SHA256SUMS','unpacked_acceptance.json','unpacked_tests.xml','unpacked_tests.log'):continue
        if p.suffix not in ('.py','.json','.txt','.md','.sh','.diff','.patch','.xml','.log','.npz'):continue
        if p.suffix=='.npz' and p!=R/'package/data/IMDB.npz':raise ValueError('unexpected dataset/result array '+str(p))
        files.append(p)
    checks=''.join(f'{sha(p)}  {p.relative_to(R)}\n' for p in sorted(files));archive=R/'dihug_imdb_role_pair_gpu_v1.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,'imdb_role_pair_gpu_v1/'+str(p.relative_to(R)))
        z.writestr('imdb_role_pair_gpu_v1/SHA256SUMS',checks)
    (R/'SHA256SUMS').write_text(f'{sha(archive)}  {archive.name}\n')
    receipt={'baseline':'f4d410cc37a76b065634eadb27a5421d53f638a0','archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size,
        'files':len(files)+1,'package_manifest_sha256':sha(R/'package/manifest.json'),'notebook_GPU_preflight_executed':False,'formal_training_executed':False,'weights_in_archive':False}
    (R/'release_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
if __name__=='__main__':main()
