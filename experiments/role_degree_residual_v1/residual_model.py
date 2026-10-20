"""Explicit soft role degrees/noise-aligned residuals inside the joint MLP.

No degree target per node is supplied: soft targets come from the existing
anonymous allocation readout. No output-logit controller or graph repair.
"""
import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
sys.path.insert(0, str(Path(__file__).resolve().parent.parent/"role_specialization_v1"))
from capacity_readout import CapacityDenoiser
from role_condition.model import capacities

def degree_inputs(current, soft, capacity, prior, alpha):
    """Bounded inputs; exact zero for structurally impossible dimensions."""
    current, soft, capacity = (np.asarray(x, dtype=np.float64) for x in (current, soft, capacity))
    if current.shape != soft.shape or capacity.shape != soft.shape:
        raise ValueError("degree feature shape mismatch")
    if not 0 <= alpha <= 1 or not all(np.isfinite(x).all() for x in (current, soft, capacity, prior)):
        raise ValueError("nonfinite degree features")
    if (capacity < 0).any() or (current < 0).any() or (current > capacity).any():
        raise ValueError("observed degrees outside capacity")
    if (soft < -1e-10).any() or (soft > capacity+1e-10).any():
        raise ValueError("soft degrees outside capacity")
    noisy = alpha*soft+(1-alpha)*capacity*prior
    clean = np.divide(soft, capacity, out=np.zeros_like(soft), where=capacity > 0)
    residual = np.divide(current-noisy, capacity, out=np.zeros_like(soft), where=capacity > 0)
    return np.concatenate([clean, residual], axis=-1)

class DegreeResidualDenoiser(CapacityDenoiser):
    architecture_version = "capacity_role_noise_aligned_degree_joint_v1"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Two R-vectors per node, symmetric mean+absolute difference => 4R.
        # Zero init ensures matched initial logits, despite the added inputs.
        self.degree_fusion = nn.Linear(4*self.n_families, self.joint_head[0].out_features, bias=False)
        nn.init.zeros_(self.degree_fusion.weight)

    def node_condition(self, ctx, t):
        value = super().node_condition(ctx, t)
        diagnostic = ctx.condition_diagnostic
        if "explicit_degree_inputs" not in diagnostic:
            types = ctx.types
            soft = np.zeros((len(types), self.n_families))
            for item in diagnostic["allocation"]:
                typ = item["type"]
                roles = np.array([v for v, count in self._target.roles[typ]])
                soft[types == typ] = item["attention"] @ roles
            cap = capacities(types, self.family_pairs)[types]
            step = int(round(float(t)*self.branch.T))
            f = degree_inputs(diagnostic["degree"], soft, cap,
                              self.prior.detach().cpu().numpy(), self.branch.alpha[step])
            diagnostic["soft_target_degree"] = soft
            diagnostic["explicit_degree_inputs"] = torch.as_tensor(f, dtype=ctx.x.dtype, device=ctx.device)
        return value

    def readout(self, features, role_pairs, degree_pairs):
        hidden = (self.joint_head[0](features)+self.role_fusion(role_pairs)
                  + self.degree_fusion(degree_pairs))
        return self.joint_head[2](self.joint_head[1](hidden))

    def joint_predict(self, ctx, graph, time, mask_cn=False, return_features=False):
        # Same production joint readout fields as LateConditionalDenoiser.
        # Only extra input is degree_pairs before the existing nonlinearity.
        self.node_condition(ctx, time)
        result, hidden = self.backbone(ctx.x, graph.attrs, graph.edges,
            ctx.x.new_tensor([[float(time)]]), ctx.batch, return_hidden=True,
            node_residual=ctx.node_residual)
        index = graph.query_index; family = graph.query_families; query = graph.edges[:, index]
        left, right = hidden["node"][query[0]], hidden["node"][query[1]]
        old_logits = torch.stack([result.edge_attr[index, 0], result.edge_attr[index, family+1]], -1)
        cn = ctx.x.new_zeros((len(index), self.cn_feature_width))
        if self.common_neighbor is not None and not mask_cn:
            cn = self.common_neighbor.aggregate(ctx, query, family, time)
        state = (graph.states[index] > 0).long()
        fields = [hidden["edge"][index], (left+right)*.5, (left-right).abs(),
            hidden["global_state"].expand(len(index), -1), ctx.x.new_full((len(index), 1), float(time)),
            ctx.global_statistics.expand(len(index), -1),
            torch.nn.functional.one_hot(family, self.n_families).float(),
            torch.nn.functional.one_hot(state, 2).float(), old_logits, cn]
        features = torch.cat(fields, -1)
        pairs = self.endpoint_roles(ctx.condition_diagnostic["role_context"], query)
        degree_pairs = self.endpoint_roles(ctx.condition_diagnostic["explicit_degree_inputs"], query)
        logits = self.readout(features, pairs, degree_pairs)
        if not torch.isfinite(features).all() or not torch.isfinite(logits).all():
            raise FloatingPointError("nonfinite explicit degree joint fusion")
        return (logits, features) if return_features else logits
