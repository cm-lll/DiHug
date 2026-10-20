"""Independent unpack, exact manifest and complete test-set verification."""
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
root=Path(__file__).resolve().parent
archive=root/"role_error_recovery_v1_review.zip"
receipt=json.loads((root/"release_receipt.json").read_text())
assert hashlib.sha256(archive.read_bytes()).hexdigest()==receipt["sha256"]
start=time.monotonic();dest=Path(tempfile.mkdtemp(prefix="role-recovery-review-"))
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None;z.extractall(dest)
package=dest/"role_error_recovery_v1_review";n=0
for line in (package/"SHA256SUMS").read_text().splitlines():
    h,name=line.split("  ",1);assert hashlib.sha256((package/name).read_bytes()).hexdigest()==h;n+=1
assert n==receipt["payload_files"]
env=dict(os.environ,OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",CUDA_VISIBLE_DEVICES="",PYTEST_ADDOPTS="",
         PYTHONPATH=str(package/"role_condition_diagnosis_v1/baseline"))
commands=[
 [sys.executable,str(package/"role_error_recovery_v1/diagnostic.py"),"--help"],
 [sys.executable,"-m","pytest","-q",str(package/"role_condition_diagnosis_v1/baseline/tests"),
  str(package/"role_condition_diagnosis_v1/test_late_fusion.py"),str(package/"role_specialization_v1/test_capacity.py"),
  str(package/"role_error_recovery_v1/test_diagnostic.py"),"--junitxml="+str(root/"unpacked_tests.xml")]]
for i,cmd in enumerate(commands):
    with (root/f"unpacked_{i}.log").open("w") as log:
        r=subprocess.run(cmd,cwd=package,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=180)
    assert r.returncode==0
old=ET.parse(package/"results/full_tests.xml");new=ET.parse(root/"unpacked_tests.xml")
def ids(tree):
    # Only remove the known repository packaging prefix. Retain the complete
    # module, function and parametrization identity; no count-only acceptance.
    def canonical(name):
        return name[len("experiments."):] if name.startswith("experiments.") else name
    return sorted((canonical(r.get("classname")),r.get("name")) for r in tree.findall(".//testcase"))
assert ids(old)==ids(new) and len(ids(new))==98
assert not new.findall(".//failure")+new.findall(".//error")+new.findall(".//skipped")
result=dict(status="passed",sha256=receipt["sha256"],payload_files=n,entry_help="passed",tests=98,
    same_exact_test_ids=True,failures=0,skipped=0,wall_seconds=time.monotonic()-start,unpacked_directory=str(package))
(root/"unpacked_acceptance.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result))
