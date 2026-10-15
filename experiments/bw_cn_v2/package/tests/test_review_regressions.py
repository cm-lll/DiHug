"""Nonzero CN diagnostic readout and finalization fault regressions (CPU only)."""
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
import torch
from hsgd.bw_preflight import common as c
from hsgd.cn_checks import small_model
from hsgd.runtime import predict
from hsgd.perf_checks import compare_float,old_details
from hsgd.computational import FrozenTypedContext,legacy_arrays,legacy_transfer

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("empty",[False,True])
def test_legacy_diagnostics_include_nonzero_cn_increment(empty):
    m,case,_=small_model();m.eval()
    with torch.no_grad():m.common_neighbor.readout.weight.fill_(.3)
    query=np.empty((2,0),dtype=np.int64) if empty else np.array([[0,0],[1,4]])
    family=case["space"].encode(query)[0]
    args=(m,case["space"].types,case["clean"],case["space"].encode(case["clean"])[0],query,family,.5)
    with torch.no_grad():
        logits,p=predict(*args,"legacy",details=True)
        expected,ep=predict(*args,"upstream",details=True)
        ctx=FrozenTypedContext(m,*args[1:4],state_version="review")
        old_graph=legacy_transfer(ctx,legacy_arrays(ctx,query,family,"review"))
        other,op=old_details(*args,old_graph)
        compare_float(logits,expected,"logits");compare_float(p,ep,"p0")
        compare_float(other,expected,"logits");compare_float(op,ep,"p0")
        pi=m.prior[torch.tensor(family)]
        from_logits=logits.softmax(-1)[:,1]
        from_logits=torch.where(pi==0,0.,torch.where(pi==1,1.,from_logits))
        compare_float(p,from_logits,"p0")
        if not empty:
            graph=ctx.assemble(ctx.transfer_query(ctx.prepare_query(query,family,"review"),"review"),"review")
            delta=m.common_neighbor(ctx,graph,.5)
            assert delta.abs().max()>.01  # zero-init must NOT mask this regression
            assert not torch.allclose(logits[:,1]-delta,logits[:,1],atol=1e-5,rtol=0)
        ctx.close()


def call(out,mode,expected=0):
    p=subprocess.run([sys.executable,str(ROOT/"tests/finalization_worker.py"),str(out),mode],
        cwd=ROOT,env=dict(os.environ,OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1"),
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=60)
    assert p.returncode==expected,p.stdout[-6000:]


def data(out):
    pointer=json.loads((out/"chain_latest.json").read_text())
    return pointer,c.trusted_load(out/pointer["path"])


@pytest.mark.parametrize("mode",["crash_after_export","stop_before_export","evaluation_failure","pending_evaluation"])
def test_finalization_resume_does_not_resample(tmp_path,mode):
    out=tmp_path/mode;call(out,mode,86 if mode=="crash_after_export" else 0)
    before,ck=data(out)
    assert ck["remaining_time"]==0 and ck["completed_reverse_steps"]==4
    if mode=="evaluation_failure":
        assert json.loads((out/"summary.json").read_text())["status"]=="incomplete_evaluation_or_integrity"
    exists=(out/"graph.npz").exists()
    old=(out/"graph.npz").read_bytes() if exists else None
    call(out,"resume_finish")
    summary=json.loads((out/"summary.json").read_text())
    assert summary["status"]=="completed" and summary["evaluator_complete"] is True
    assert summary["reused_final_graph"]==exists
    assert data(out)[0]==before  # no checkpoint or reverse step rewritten
    if exists:assert (out/"graph.npz").read_bytes()==old
    call(out,"reject_already_complete")


@pytest.mark.parametrize("field",["source_model_sha256","seed","edges","relation_degree",
                                   "sampling_binding_sha256","final_state_checkpoint_sha256"])
def test_existing_graph_corruption_blocks_resume_without_overwrite(tmp_path,field):
    out=tmp_path/field;call(out,"pending_evaluation")
    with np.load(out/"graph.npz",allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
    if field in ("source_model_sha256","sampling_binding_sha256","final_state_checkpoint_sha256"):
        arrays[field]=np.asarray("0"*64)
    elif field=="seed":arrays[field]=arrays[field]+1
    else:
        arrays[field]=arrays[field].copy();arrays[field].flat[0]+=1
    with (out/"graph.npz").open("wb") as f:np.savez_compressed(f,**arrays)
    corrupt=(out/"graph.npz").read_bytes();pointer=data(out)[0]
    call(out,"resume_reject_graph")
    assert (out/"graph.npz").read_bytes()==corrupt and data(out)[0]==pointer
    assert json.loads((out/"summary.json").read_text())["status"]=="failed"


def test_corrupt_terminal_checkpoint_blocks_resume(tmp_path):
    out=tmp_path/"checkpoint";call(out,"pending_evaluation")
    pointer,_=data(out);p=out/pointer["path"];p.write_bytes(p.read_bytes()+b"corrupt")
    call(out,"resume_reject_checkpoint")
    assert "checkpoint hash mismatch" in json.loads((out/"summary.json").read_text())["reason"]


def test_post_evaluation_status_requires_complete_steps(tmp_path):
    from hsgd.cn_entry import prepare_sample_resume
    out=tmp_path/"bad_steps";call(out,"pending_evaluation")
    summary=json.loads((out/"summary.json").read_text())
    summary.update(status="incomplete_evaluation_or_integrity",completed_reverse_steps=3)
    (out/"summary.json").write_text(json.dumps(summary));original=(out/"summary.json").read_bytes()
    with pytest.raises(RuntimeError,match="complete final state"):prepare_sample_resume(out,4)
    assert (out/"summary.json").read_bytes()==original
