"""Fault injection only; no formal weights or scientific graph generation."""
from pathlib import Path
import json
import os
import sys
from unittest.mock import patch
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hsgd.cn_checks import small_model
from hsgd import cn_sampling as s
from hsgd.cn_entry import prepare_sample_resume,finish_sample_evaluation
from hsgd.learning_budget import model_fingerprint

m,case,cfg=small_model();m.eval();cfg.update(T=4,rounds=3)
with torch.no_grad():m.common_neighbor.readout.weight.fill_(.013)
out=Path(sys.argv[1]);mode=sys.argv[2]
receipt=dict(formal_training=False,completed_updates=0,environment=dict(utc="fixture"),
    model_sha256=model_fingerprint(m.state_dict()),final_checkpoint="engineering fixture only",
    source_package=dict(path=str(s.ROOT),files={}),run=str(out),run_evidence={})
provenance=s.ROOT/"data/IMDB_preprocessing_provenance.json"
if mode.startswith("resume_") or mode=="reject_already_complete":
    if mode=="reject_already_complete":
        try:prepare_sample_resume(out,4)
        except RuntimeError as exc:
            assert "already completed" in str(exc);raise SystemExit(0)
        raise AssertionError("fully evaluated sample accepted twice")
    prepare_sample_resume(out,4)
    with patch("hsgd.runtime.reverse_step",side_effect=AssertionError("must not resample during finalization")):
        code=s.chain(m,case,cfg,receipt,out,1613,"cpu",resume=True)
    if mode=="resume_finish":
        assert code==0
        finish_sample_evaluation(case,receipt,out,provenance)
    else:
        assert code!=0
        summary=s.read(out/"summary.json")
        assert summary["status"]=="failed"
        if mode=="resume_reject_graph":assert "final graph" in summary["reason"]
else:
    out.mkdir()
    if mode=="crash_after_export":
        original=s.finalize_graph
        def crash(*a,**kw):
            original(*a,**kw)
            os._exit(86)  # real process exit: no terminal summary or cleanup
        with patch.object(s,"finalize_graph",side_effect=crash):
            s.chain(m,case,cfg,receipt,out,1613,"cpu")
    else:
        assert s.chain(m,case,cfg,receipt,out,1613,"cpu",stop_after=4 if mode=="stop_before_export" else None)==0
        if mode=="evaluation_failure":
            with patch("hsgd.cn_metrics.evaluate",side_effect=RuntimeError("injected post-evaluation failure")):
                try:finish_sample_evaluation(case,receipt,out,provenance)
                except RuntimeError as exc:assert "injected" in str(exc)
                else:raise AssertionError("post-evaluation failure hidden")
        else:assert mode in ("stop_before_export","pending_evaluation")
