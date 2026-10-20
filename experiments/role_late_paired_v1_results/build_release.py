"""Archive completed evidence and frozen source; never invoke the experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
import time


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--run",type=Path,required=True);a=ap.parse_args()
    root=Path(__file__).resolve().parent;experiments=root.parent
    assert json.loads((a.run/"budget_receipt.json").read_text())["status"]=="completed"
    assert json.loads((a.run/"audit.json").read_text())["status"]=="completed"
    ledger=a.run.with_name(a.run.name+".budget.json");budget=json.loads(ledger.read_text())
    if time.time()>budget["hard_cutoff_epoch"]:raise TimeoutError("original execution/acceptance budget expired")
    evidence=root/"evidence";evidence.mkdir(exist_ok=False)
    weight_hashes=json.loads((a.run/"weight_hashes.json").read_text())
    for path,h in weight_hashes.items():assert sha(a.run/path)==h
    for p in a.run.rglob("*"):
        if not p.is_file() or p.suffix==".pt":continue
        dest=evidence/p.relative_to(a.run);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    shutil.copy2(ledger,evidence/"activation_budget.json")
    (root/"weight_storage.json").write_text(json.dumps(dict(
        actual_weights_saved=True,local_run=str(a.run),files=weight_hashes,
        public_delivery="all 12 actual initial/final files remain in persistent local run; GitHub publishes SHA256 receipts, not repeated weight bytes",
        total_bytes=sum((a.run/n).stat().st_size for n in weight_hashes)),indent=2)+"\n")
    # Standalone directory layout matches the imports/source manifest.
    package=root/"review_bundle";package.mkdir()
    for name in ["role_late_paired_v1"]:
        shutil.copytree(experiments/name,package/name,ignore=shutil.ignore_patterns("__pycache__",".pytest_cache","*.pyc"))
    previous=package/"role_condition_diagnosis_v1";previous.mkdir()
    shutil.copytree(experiments/"role_condition_diagnosis_v1/baseline",previous/"baseline",
                   ignore=shutil.ignore_patterns("__pycache__",".pytest_cache","*.pyc"))
    for name in ["late_fusion.py","test_late_fusion.py"]:
        shutil.copy2(experiments/"role_condition_diagnosis_v1"/name,previous/name)
    shutil.copytree(evidence,package/"results")
    shutil.copy2(root/"weight_storage.json",package/"weight_storage.json")
    manifest={str(p.relative_to(package)):sha(p) for p in sorted(package.rglob("*")) if p.is_file()}
    (package/"SHA256SUMS").write_text("".join(f"{h}  {p}\n" for p,h in manifest.items()))
    archive=root/"role_late_paired_v1_review.zip"
    with zipfile.ZipFile(archive,"x",zipfile.ZIP_DEFLATED) as z:
        for p in sorted(package.rglob("*")):
            if p.is_file():z.write(p,"role_late_paired_v1_review/"+str(p.relative_to(package)))
    receipt=dict(zip_sha256=sha(archive),zip_bytes=archive.stat().st_size,files=len(manifest)+1,
         original_activation_epoch=budget["activation_epoch"],saved_epoch=time.time(),
         cumulative_seconds_to_archive=time.time()-budget["activation_epoch"],no_reset=True)
    assert receipt["cumulative_seconds_to_archive"]<3600
    (root/"release_receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt))
if __name__=="__main__":main()
