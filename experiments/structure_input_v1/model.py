"""A late-CN plus typed closure/component inputs, CPU reference backend.

Import with the frozen science package on sys.path, as in the CPU test.
No loss, transition, optimizer, EMA or production factory is replaced.
"""
import torch
from torch import nn
from model_variants import StructuralDenoiser
from .features_cpu import extract


class ZeroProjection(nn.Module):
    """Bias-free projection whose construction consumes no random numbers."""
    def __init__(self, inputs, outputs):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(outputs, inputs))

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight)


class StructureInputA(StructuralDenoiser):
    architecture_version = 'a_late_cn_typed_closure_component_input_v1'

    def __init__(self, case, cfg):
        super().__init__(case, cfg, 'A')
        channels = self.n_types * (self.n_types + 1) // 2
        self.structure_node_projection = ZeroProjection(channels + 1 + self.n_types,
                                                         cfg['model']['width'])
        self.structure_pair_projection = ZeroProjection(1, cfg['model']['edge_width'])

    @classmethod
    def from_initial_A(cls, source, case, cfg, *, completed_updates):
        """Strict shared-state transfer; never a trained-checkpoint upgrade.

        Caller must independently verify the supplied boundary/asset binding.
        This routine validates declared update count, architecture and tensors;
        it cannot prove a checkpoint's historical provenance by itself.
        """
        if completed_updates != 0 or type(source) is not StructuralDenoiser or source.arm != 'A':
            raise ValueError('requires the declared untrained ordinary A state')
        if hasattr(source, 'cn_edge_projection'):
            raise ValueError('early CN is not this baseline')
        # Tensor shapes cannot establish relation semantics. Validate the ordered
        # schema and exact relation prior before copying any source tensors.
        pairs = tuple(tuple(int(v) for v in p) for p in case['space'].pairs)
        if (source.n_types != case['n_types'] or source.n_families != case['n_families']
                or tuple(source.family_pairs) != pairs):
            raise ValueError('ordered schema or type/relation inventory differs')
        prior = torch.as_tensor(case['prior'], dtype=source.prior.dtype, device=source.prior.device)
        if not torch.equal(source.prior, prior):
            raise ValueError('relation prior/support differs')
        with torch.random.fork_rng(devices=[]):
            target = cls(case, cfg)
        state = target.state_dict()
        original = source.state_dict()
        extras = {'structure_node_projection.weight', 'structure_pair_projection.weight'}
        if set(state) - extras != set(original):
            raise ValueError('baseline state keys differ')
        for name, value in original.items():
            if state[name].shape != value.shape or state[name].dtype != value.dtype:
                raise ValueError('baseline shape/dtype differs: ' + name)
            state[name] = value.detach().clone()
        target.load_state_dict(state, strict=True)
        target.to(next(source.parameters()).device)
        target.train(source.training)
        return target

    def _structure_inputs(self, ctx):
        ctx.check(ctx.state_version)
        if not hasattr(ctx, '_structure_input_v1'):
            # Read only the immutable physical Xt. Candidate query edges are
            # deliberately absent. Reference backend does not yet reuse CN wedges.
            values = extract(ctx.types.tolist(), [tuple(x) for x in ctx.old.T.tolist()],
                             ctx.of.tolist(), ctx.pairs.tolist(), self.n_types)
            node = torch.tensor(values['node_features'], dtype=ctx.x.dtype, device=ctx.device)
            ids = torch.tensor(values['component_ids'], dtype=torch.long, device=ctx.device)
            if not torch.isfinite(node).all():
                raise FloatingPointError('nonfinite structure inputs')
            ctx._structure_input_v1 = (node, ids)
            ctx._guards.extend([(node, node._version), (ids, ids._version)])
        return ctx._structure_input_v1

    def structure_node_residual(self, ctx):
        node, _ = self._structure_inputs(ctx)
        return self.structure_node_projection(node)

    def structure_pair_residual(self, ctx, query):
        _, ids = self._structure_inputs(ctx)
        same = (ids[query[0]] == ids[query[1]]).to(ctx.x.dtype).unsqueeze(-1)
        return self.structure_pair_projection(same)
