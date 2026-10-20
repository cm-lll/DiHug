"""Independent archive verification; bounded by the original experiment ledger."""
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

root = Path(__file__).resolve().parent
ledger = json.loads((root / "evidence/activation_budget.json").read_text())
start = time.time()
destination = Path(tempfile.mkdtemp(prefix="role-late-review-"))
archive = root / "role_late_paired_v1_review.zip"
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    z.extractall(destination)
package = destination / "role_late_paired_v1_review"
checked = 0
for line in (package / "SHA256SUMS").read_text().splitlines():
    digest, name = line.split("  ", 1)
    assert hashlib.sha256((package / name).read_bytes()).hexdigest() == digest
    checked += 1
env = dict(os.environ, PYTEST_ADDOPTS="", OMP_NUM_THREADS="1",
           OPENBLAS_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="",
           PYTHONPATH=str(package / "role_condition_diagnosis_v1/baseline"))
commands = [
    [sys.executable, str(package / "role_late_paired_v1/run.py"), "--help"],
    [sys.executable, "-m", "pytest", "-q",
     str(package / "role_condition_diagnosis_v1/baseline/tests"),
     str(package / "role_condition_diagnosis_v1/test_late_fusion.py"),
     str(package / "role_late_paired_v1/test_pairing.py"),
     "--junitxml=" + str(root / "unpacked_tests.xml")],
]
records = []
for i, command in enumerate(commands):
    remaining = ledger["hard_cutoff_epoch"] - time.time()
    assert remaining > 0
    with (root / f"unpacked_{i}.log").open("w") as log:
        result = subprocess.run(command, cwd=package, env=env, stdout=log,
                                stderr=subprocess.STDOUT, timeout=remaining)
    records.append(dict(command=command, exit_code=result.returncode))
    assert result.returncode == 0
tree = ET.parse(root / "unpacked_tests.xml")
cases = tree.findall(".//testcase")
assert len(cases) == 81
assert not tree.findall(".//skipped") + tree.findall(".//failure") + tree.findall(".//error")
receipt = dict(status="passed", archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
               files_checked=checked, tests=81, skipped=0, failed=0, commands=records,
               extracted_to=str(destination), wall_seconds=time.time()-start,
               cumulative_seconds=time.time()-ledger["activation_epoch"],
               original_deadline_epoch=ledger["hard_cutoff_epoch"], no_reset=True)
assert receipt["cumulative_seconds"] < 3600
(root / "unpacked_acceptance.json").write_text(json.dumps(receipt, indent=2)+"\n")
print(json.dumps(receipt))
