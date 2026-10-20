import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
import zipfile
root=Path(__file__).resolve().parent;archive=root/"role_robust_training_v1_review.zip"
receipt=json.loads((root/"release_receipt.json").read_text());assert hashlib.sha256(archive.read_bytes()).hexdigest()==receipt["sha256"]
start=time.monotonic();destination=Path(tempfile.mkdtemp(prefix="role-robust-review-"))
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None;z.extractall(destination)
pkg=destination/"role_robust_training_v1_review";count=0
for line in (pkg/"SHA256SUMS").read_text().splitlines():
    h,n=line.split("  ",1);assert hashlib.sha256((pkg/n).read_bytes()).hexdigest()==h;count+=1
assert count==receipt["payload_files"]
env=dict(os.environ,OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",CUDA_VISIBLE_DEVICES="",PYTEST_ADDOPTS="",
         PYTHONPATH=str(pkg/"role_condition_diagnosis_v1/baseline"))
commands=[
 [sys.executable,str(pkg/"role_robust_training_v1/run.py"),"--help"],
 [sys.executable,str(pkg/"role_robust_training_v1/worker.py"),"--help"],
 [sys.executable,"-c","import sys,json;sys.path.insert(0,'role_robust_training_v1');from core import file_manifest;assert file_manifest()==json.load(open('results/frozen/source_manifest.json'));print('frozen source binding exact')"],
 [sys.executable,"-m","pytest","-q",str(pkg/"role_condition_diagnosis_v1/baseline/tests"),
  str(pkg/"role_condition_diagnosis_v1/test_late_fusion.py"),str(pkg/"role_specialization_v1/test_capacity.py"),
  str(pkg/"role_error_recovery_v1/test_diagnostic.py"),str(pkg/"role_robust_training_v1/test_robust.py"),
  "--junitxml="+str(root/"unpacked_tests.xml")]]
for i,cmd in enumerate(commands):
    with (root/f"unpacked_{i}.log").open("w") as f:r=subprocess.run(cmd,cwd=pkg,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=180)
    assert r.returncode==0,(i,r.returncode)
def ids(tree):
    def canon(c):return c[len("experiments."):] if c.startswith("experiments.") else c
    return sorted((canon(x.get("classname")),x.get("name")) for x in tree.findall(".//testcase"))
old=ET.parse(pkg/"results/full_tests.xml");new=ET.parse(root/"unpacked_tests.xml")
assert ids(old)==ids(new) and len(ids(new))==126
assert not new.findall(".//failure")+new.findall(".//error")+new.findall(".//skipped")
result=dict(status="passed",payload_files=count,sha256=receipt["sha256"],tests=126,failures=0,skipped=0,
    exact_test_set=True,frozen_source_binding="exact",parent_and_worker_entry_help="passed",
    wall_seconds=time.monotonic()-start,unpacked_directory=str(pkg),learning_window_activated=False)
(root/"unpacked_acceptance.json").write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result))
