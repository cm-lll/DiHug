"""Only the frozen late-CN constructors, with no early residual installer."""
from common_run import SCI
import torch,hashlib
from model_variants import StructuralDenoiser
def tensor_sha(v):return hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def make_model(case,cfg,configuration,basis):
 if configuration not in ['A_late','D_late','E_late']:raise ValueError('fixed late arms only')
 m=StructuralDenoiser(case,cfg,configuration[0],basis if configuration=='E_late' else None)
 if hasattr(m,'cn_edge_projection') or m.joint_head[0].in_features!=776:raise ValueError('late-only path')
 return m
