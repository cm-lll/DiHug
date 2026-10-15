"""Finite engineering fixture for native CN chain checkpoint tests only."""
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hsgd.cn_checks import small_model
from hsgd.cn_sampling import chain
from hsgd.learning_budget import model_fingerprint
m,case,cfg=small_model();m.eval();cfg.update(T=4,rounds=3)
with torch.no_grad():m.common_neighbor.readout.weight.fill_(.013)
receipt=dict(formal_training=False,completed_updates=0,environment=dict(utc="fixture"),
    model_sha256=model_fingerprint(m.state_dict()),final_checkpoint="fixed engineering model, no scientific weights")
out=Path(sys.argv[1]);mode=sys.argv[2]
if mode!="resume":out.mkdir()
raise SystemExit(chain(m,case,cfg,receipt,out,1613,"cpu",resume=mode=="resume",
    stop_after=2 if mode=="split" else None,interrupt_block=1 if mode=="interrupt" else None))
