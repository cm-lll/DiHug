"""Nonzero CN weights, complete CPU engineering chain and new-process resume."""
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from hsgd.bw_preflight import common as c
from hsgd.perf_checks import compare_rng

ROOT=Path(__file__).resolve().parents[1]


def call(out,mode):
    r=subprocess.run([sys.executable,str(ROOT/"tests/cn_sampling_worker.py"),str(out),mode],cwd=ROOT,
        env=dict(os.environ,OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1"),capture_output=True,text=True,timeout=60)
    assert r.returncode==0,r.stdout[-4000:]+r.stderr[-4000:]


def load(out):
    p=json.loads((out/"chain_latest.json").read_text());return c.trusted_load(out/p["path"])


def test_nonzero_cn_complete_vs_split_restore(tmp_path):
    full=tmp_path/"full";split=tmp_path/"split"
    call(full,"full");call(split,"split")
    before=json.loads((split/"summary.json").read_text())
    assert before["status"]=="stopped" and before["completed_reverse_steps"]==2
    call(split,"resume")
    a,b=load(full),load(split)
    assert a["remaining_time"]==b["remaining_time"]==0
    assert np.array_equal(a["packed_state"],b["packed_state"])
    compare_rng(a["rng"],b["rng"])
    with np.load(full/"graph.npz") as x,np.load(split/"graph.npz") as y:assert np.array_equal(x["edges"],y["edges"])
    assert json.loads((split/"summary.json").read_text())["scientific_output_count"]==0


def test_nonzero_cn_partial_block_transaction(tmp_path):
    out=tmp_path/"interrupted"
    call(out,"interrupt")
    before=load(out)
    assert before["completed_reverse_steps"]==0 and before["remaining_time"]==4
    assert not (out/"graph.npz").exists()
    call(out,"resume")
    assert json.loads((out/"summary.json").read_text())["status"]=="completed"
