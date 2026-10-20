"""Package reviewed source and preparation evidence; no inference calls."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--results",type=Path,required=True);a=ap.parse_args()
    root=Path(__file__).resolve().parent;exp=root.parent
    for stage in ("frozen","probe"):
        r=json.loads((a.results/stage/"worker_receipt.json").read_text())
        assert r["status"]=="completed" and not r["learning_panel_activated"] and r["scientific_updates"]==0
    assert not (a.results/"learning_run01").exists()
    evidence=root/"evidence";shutil.copytree(a.results,evidence)
    patch=subprocess.check_output(["git","diff","72c764815f1678bef76c7dc599ca978848f8e7a9","HEAD","--","experiments/role_robust_training_v1"],cwd=exp.parent)
    assert patch.startswith(b"diff --git")
    (root/"source_vs_72c7648.patch").write_bytes(patch)
    bundle=root/"review_bundle";bundle.mkdir();ignore=shutil.ignore_patterns("__pycache__",".pytest_cache","*.pyc")
    for name in ("role_robust_training_v1","role_specialization_v1","role_late_paired_v1","role_error_recovery_v1"):
        shutil.copytree(exp/name,bundle/name,ignore=ignore)
    prev=bundle/"role_condition_diagnosis_v1";prev.mkdir()
    shutil.copytree(exp/"role_condition_diagnosis_v1/baseline",prev/"baseline",ignore=ignore)
    for name in ("diagnose.py","late_fusion.py","test_late_fusion.py"):
        shutil.copy2(exp/"role_condition_diagnosis_v1"/name,prev/name)
    old=exp/"role_specialization_v1_results/evidence/learning_run01"
    dest=bundle/"role_specialization_v1_results/evidence/learning_run01";dest.mkdir(parents=True)
    for name in ("resolved_config.json","weight_hashes.json","source_manifest.json","environment.json"):
        shutil.copy2(old/name,dest/name)
    panel=bundle/"role_error_recovery_v1_results/evidence/archived_panel";panel.mkdir(parents=True)
    shutil.copy2(exp/"role_error_recovery_v1_results/evidence/archived_panel/panel.json",panel/"panel.json")
    shutil.copytree(evidence,bundle/"results")
    shutil.copy2(root/"README.md",bundle/"README.md");shutil.copy2(root/"source_vs_72c7648.patch",bundle/"source_vs_72c7648.patch")
    manifest={str(p.relative_to(bundle)):sha(p) for p in sorted(bundle.rglob("*")) if p.is_file()}
    assert not any(n.endswith(".pt") for n in manifest)
    (bundle/"SHA256SUMS").write_text("".join(f"{h}  {n}\n" for n,h in manifest.items()))
    path=root/"role_robust_training_v1_review.zip"
    with zipfile.ZipFile(path,"x",zipfile.ZIP_DEFLATED) as z:
        for p in sorted(bundle.rglob("*")):
            if p.is_file():z.write(p,"role_robust_training_v1_review/"+str(p.relative_to(bundle)))
    receipt=dict(sha256=sha(path),bytes=path.stat().st_size,payload_files=len(manifest),
                 learning_window_activated=False,scientific_source_commit="38a4f437d00f8066c7123dd0bbb5c9cfc6f674ed")
    (root/"release_receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
    (root/"DELIVERY_SHA256SUMS").write_text(f"{sha(path)}  {path.name}\n{sha(root/'source_vs_72c7648.patch')}  source_vs_72c7648.patch\n")
    print(json.dumps(receipt))
if __name__=="__main__":main()
