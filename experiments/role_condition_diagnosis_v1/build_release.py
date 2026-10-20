"""Package only; never train or sample. Run after evidence has been copied."""
import hashlib
import json
from pathlib import Path
import zipfile


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    root=Path(__file__).resolve().parent
    archive=root/"dihug_role_condition_diagnosis_v1.zip"
    if archive.exists():raise FileExistsError("preserve existing ZIP")
    excluded={"SHA256SUMS","release_manifest.json",archive.name,"unpacked_acceptance.json"}
    files=sorted(p for p in root.rglob("*") if p.is_file() and not any(
        x in p.parts for x in ("__pycache__",".pytest_cache",".git"))
        and p.name not in excluded and p.suffix not in (".pyc",".pt"))
    manifest={str(p.relative_to(root)):sha(p) for p in files}
    (root/"release_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    manifest["release_manifest.json"]=sha(root/"release_manifest.json")
    (root/"SHA256SUMS").write_text("".join(f"{h}  {n}\n" for n,h in sorted(manifest.items())))
    with zipfile.ZipFile(archive,"x",zipfile.ZIP_DEFLATED) as z:
        for name in [*manifest,"SHA256SUMS"]:z.write(root/name,root.name+"/"+name)
    print(json.dumps(dict(zip=archive.name,sha256=sha(archive),bytes=archive.stat().st_size,files=len(manifest)+1)))
if __name__=="__main__":main()
