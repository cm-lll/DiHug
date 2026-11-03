"""Only the frozen late-CN constructors, with no early residual installer."""
from common_run import SCI
import torch,hashlib
from model_variants import StructuralDenoiser
def tensor_sha(v):return hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def make_model(case,cfg,configuration,basis):
 if configuration not in ['O_late','A_late']:raise ValueError('fixed late arms only')
 m=StructuralDenoiser(case,cfg,configuration[0],basis if configuration=='E_late' else None)
 if hasattr(m,'cn_edge_projection') or m.joint_head[0].in_features!=(739 if configuration=='O_late' else 776):raise ValueError('late-only path')
 if configuration=='O_late' and (m.common_neighbor is not None or hasattr(m,'refex_projection') or hasattr(m,'role_projection')):raise ValueError('O architecture identity')
 return m
