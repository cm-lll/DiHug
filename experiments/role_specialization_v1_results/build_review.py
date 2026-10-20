"""Publishable local evidence/source bundle; no model or sampling invocation."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--results",type=Path,required=True);ap.add_argument("--baseline",type=Path,required=True)
    a=ap.parse_args();root=Path(__file__).resolve().parent;exp=root.parent;run=a.results/"learning_run01"
    assert json.loads((run/"audit.json").read_text())["status"]=="completed"
    ledger=json.loads((run/"activation.json").read_text())
    if time.time()>ledger["deadline_epoch"]:raise TimeoutError("original single window expired")
    evidence=root/"evidence";evidence.mkdir(exist_ok=False)
    for p in a.results.rglob("*"):
        if p.is_file() and p.suffix!=".pt":
            dest=evidence/p.relative_to(a.results);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    weights=json.loads((run/"weight_hashes.json").read_text())
    for p,h in weights.items():assert sha(run/p)==h
    (root/"weight_storage.json").write_text(json.dumps(dict(actual_initial_final_weights_saved=True,
        path=str(run),hashes=weights,public_delivery="SHA256 receipts; actual synthetic weight files retained locally"),indent=2)+"\n")
    patch=subprocess.check_output(["git","diff","f2f58a39ced8f895ebc636ad49b5ee91509f6596","HEAD","--","experiments/role_specialization_v1"],cwd=exp.parent)
    assert patch.startswith(b"diff --git ") and len(patch)>1000
    (root/"source_vs_f2f58a3.patch").write_bytes(patch)
    bundle=root/"review_bundle";bundle.mkdir(exist_ok=False)
    for folder in ["role_specialization_v1","role_late_paired_v1"]:
        shutil.copytree(exp/folder,bundle/folder,ignore=shutil.ignore_patterns("__pycache__",".pytest_cache","*.pyc"))
    previous=bundle/"role_condition_diagnosis_v1";previous.mkdir()
    shutil.copytree(exp/"role_condition_diagnosis_v1/baseline",previous/"baseline",ignore=shutil.ignore_patterns("__pycache__",".pytest_cache","*.pyc"))
    for name in ["late_fusion.py","test_late_fusion.py","diagnose.py"]:
        shutil.copy2(exp/"role_condition_diagnosis_v1"/name,previous/name)
    shutil.copytree(evidence,bundle/"results")
    shutil.copy2(root/"weight_storage.json",bundle/"weight_storage.json")
    shutil.copy2(root/"RESULTS.md",bundle/"RESULTS.md")
    tools=bundle/"review_tools";tools.mkdir()
    for name in ["summarize.py","check_joint.py"]:
        shutil.copy2(root/name,tools/name)
    # Archived comparator is supplied read-only; never represented as new samples.
    base=bundle/"baseline_evidence";base.mkdir()
    for p in a.baseline.rglob("*"):
        if p.is_file() and p.suffix!=".pt":
            dest=base/p.relative_to(a.baseline);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    manifest={str(p.relative_to(bundle)):sha(p) for p in sorted(bundle.rglob("*")) if p.is_file()}
    (bundle/"SHA256SUMS").write_text("".join(f"{h}  {name}\n" for name,h in manifest.items()))
    path=root/"role_specialization_v1_review.zip"
    with zipfile.ZipFile(path,"x",zipfile.ZIP_DEFLATED) as z:
        for p in sorted(bundle.rglob("*")):
            if p.is_file():z.write(p,"role_specialization_v1_review/"+str(p.relative_to(bundle)))
    receipt=dict(zip_sha256=sha(path),bytes=path.stat().st_size,files=len(manifest)+1,
        original_activation_epoch=ledger["activation_epoch"],cumulative_seconds=time.time()-ledger["activation_epoch"],
        no_reset=True)
    assert receipt["cumulative_seconds"]<2700
    (root/"release_receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt))
if __name__=="__main__":main()
