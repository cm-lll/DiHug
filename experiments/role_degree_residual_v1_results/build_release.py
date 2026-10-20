"""Package immutable science, all outcomes and checks; no model invocation."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
import zipfile

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def copy_evidence(source, dest):
    dest.mkdir(parents=True, exist_ok=False)
    for p in sorted(source.rglob("*")):
        if p.is_file() and p.suffix != ".pt":
            q=dest/p.relative_to(source);q.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(p,q)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--results",type=Path,required=True)
    ap.add_argument("--baseline",type=Path,required=True)
    ap.add_argument("--panel",type=Path,required=True)
    args=ap.parse_args()
    root=Path(__file__).resolve().parent;exp=root.parent
    run=args.results/"learning_run01"
    assert json.loads((run/"audit.json").read_text())["status"]=="completed"
    assert json.loads((run/"joint_independent_audit.json").read_text())["status"]=="passed"
    ledger=json.loads((run/"activation.json").read_text())
    assert time.time()<ledger["deadline_epoch"]
    evidence=root/"evidence";copy_evidence(args.results,evidence)
    frozen_panel=evidence/"input_panel";frozen_panel.mkdir()
    for name in ("panel.json","resolved_config.json"):
        shutil.copy2(args.panel/name,frozen_panel/name)
    weights=json.loads((run/"weight_hashes.json").read_text())
    assert len(weights)==6
    for p,h in weights.items():assert sha(run/p)==h
    (root/"weight_storage.json").write_text(json.dumps(dict(
        actual_initial_final_weights_saved=True,local_path=str(run),hashes=weights,
        public_delivery="hash receipts only; all six synthetic actual weight files retained locally"),indent=2)+"\n")
    patch=subprocess.check_output(["git","diff","b8e97cc42e6e835e0ea228a6666e3a22a200e035",
        "HEAD","--","experiments/role_degree_residual_v1"],cwd=exp.parent)
    assert patch.startswith(b"diff --git ") and len(patch)>1000
    (root/"source_vs_b8e97cc.patch").write_bytes(patch)
    bundle=root/"review_bundle";bundle.mkdir(exist_ok=False)
    for folder in ("role_degree_residual_v1","role_specialization_v1","role_late_paired_v1"):
        shutil.copytree(exp/folder,bundle/"experiments"/folder,
            ignore=shutil.ignore_patterns("__pycache__",".pytest_cache","*.pyc"))
    prev=bundle/"experiments/role_condition_diagnosis_v1";prev.mkdir()
    shutil.copytree(exp/"role_condition_diagnosis_v1/baseline",prev/"baseline",
        ignore=shutil.ignore_patterns("__pycache__",".pytest_cache","*.pyc"))
    for name in ("late_fusion.py","test_late_fusion.py","diagnose.py"):
        shutil.copy2(exp/"role_condition_diagnosis_v1"/name,prev/name)
    shutil.copytree(evidence,bundle/"results")
    copy_evidence(args.baseline,bundle/"baseline_evidence")
    for name in ("RESULTS.md","weight_storage.json","source_vs_b8e97cc.patch"):
        shutil.copy2(root/name,bundle/name)
    tools=bundle/"experiments/role_degree_residual_v1_results";tools.mkdir()
    for name in ("check_joint.py","replay_new.py","audit_replay.py",
                 "training_domain.py","fixed_input_response.py"):
        shutil.copy2(root/name,tools/name)
    manifest={str(p.relative_to(bundle)):sha(p) for p in sorted(bundle.rglob("*")) if p.is_file()}
    (bundle/"SHA256SUMS").write_text("".join(f"{h}  {p}\n" for p,h in manifest.items()))
    archive=root/"role_degree_residual_v1_review.zip"
    with zipfile.ZipFile(archive,"x",zipfile.ZIP_DEFLATED) as z:
        for p in sorted(bundle.rglob("*")):
            if p.is_file():z.write(p,"role_degree_residual_v1_review/"+str(p.relative_to(bundle)))
    receipt=dict(zip_sha256=sha(archive),bytes=archive.stat().st_size,payload_files=len(manifest),
        cumulative_seconds=time.time()-ledger["activation_epoch"],no_reset=True)
    assert receipt["cumulative_seconds"]<2700
    (root/"release_receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt))

if __name__=="__main__":main()
