"""Actual A class, full parameter widths, 12 synthetic nodes, CPU only."""
import copy
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('DIHUG_TEST_TMP', str(Path(__file__).resolve().parents[1]/'results/structure_input_cpu_tests/temporary'))
import test_abort_boundaries as base
from test_role_validation_boundaries import fixtures
import torch
from boundary_transactions import TrainingBoundary, rng_capture, rng_restore
from ema import ParameterEMA
from hsgd.computational import FrozenTypedContext
from model_variants import StructuralDenoiser
from experiments.structure_input_v1.model import StructureInputA

update = base.load(base.ROOT/'experiments/o_cn_matched_execute_v1/single_input.py', 'input_candidate_update')


def setup():
    rng = base.reset(1217)
    case, cfg, _ = fixtures()
    ordinary = StructuralDenoiser(case, cfg, 'A')
    before = base.stable(rng_capture(rng, 'cpu'))
    model = StructureInputA.from_initial_A(ordinary, case, cfg, completed_updates=0)
    assert before == base.stable(rng_capture(rng, 'cpu'))
    return case, cfg, ordinary, model, rng


def predict(model, case, inp):
    with torch.no_grad():
        ctx = FrozenTypedContext(model, case['space'].types, inp['edges'], inp['edge_families'], 'parity')
        try:
            _, logits, _ = ctx.predict(inp['query'], inp['f'], inp['t']/1000, 'parity', details=True)
            return logits
        finally:
            ctx.close()


class InputIntegration(unittest.TestCase):
    def test_zero_projection_identity_and_gradient_connection(self):
        case, cfg, ordinary, model, rng = setup()
        self.assertEqual(sum(p.numel() for p in model.parameters()) -
                         sum(p.numel() for p in ordinary.parameters()), 3904)
        self.assertEqual(model.joint_head[0].in_features, ordinary.joint_head[0].in_features)
        inp = update.one_input(case, cfg, 1, rng, force_t=500)
        state = rng_capture(rng, 'cpu')
        ordinary.eval(); model.eval()
        x = predict(ordinary, case, inp)
        rng_restore(state, rng, 'cpu')
        y = predict(model, case, inp)
        self.assertTrue(torch.equal(x, y))
        self.assertEqual(base.stable(state), base.stable(rng_capture(rng, 'cpu')))
        opt = torch.optim.AdamW(model.parameters(), **{k:v for k,v in cfg['optimizer'].items() if k!='type'})
        ema = ParameterEMA(model)
        tx = TrainingBoundary(model, opt, ema, rng, 'cpu', 0)
        with tx.step(1):
            update.update(model, opt, case, inp, 1000, 'cpu', ema, 0, transaction=tx)
        for layer in (model.structure_node_projection, model.structure_pair_projection):
            self.assertTrue(torch.isfinite(layer.weight.grad).all())
            self.assertGreater(float(layer.weight.grad.norm()), 0.)

    def test_cache_scope_query_independence_and_mutation_guard(self):
        case, cfg, _, model, rng = setup()
        inp = update.one_input(case, cfg, 1, rng, force_t=500)
        ctx = FrozenTypedContext(model, case['space'].types, inp['edges'], inp['edge_families'], 'x')
        original = model._structure_inputs(ctx)
        a = model.structure_pair_residual(ctx, torch.tensor(inp['query'][:, :3]))
        b = model.structure_pair_residual(ctx, torch.tensor(inp['query'][:, :7]))[:3]
        self.assertTrue(torch.equal(a, b))
        self.assertIs(original, model._structure_inputs(ctx))
        self.assertNotIn('_structure_input_v1', model.state_dict())
        original[0].add_(1.)
        with self.assertRaises(ValueError): model._structure_inputs(ctx)
        ctx.close()
        with self.assertRaises(ValueError): model._structure_inputs(ctx)

    def test_nonzero_pair_projection_reorder_partition_and_extension(self):
        case, cfg, _, model, rng = setup()
        # Components: {0,3,6}, {1,4}, plus isolated vertices. Expected flags
        # below come from this construction, not from the extractor under test.
        edges = base.np.array([(0,3), (1,4), (3,6)], dtype=base.np.int64).T
        types = case['space'].types
        pairs = [tuple(p) for p in case['space'].pairs]
        families = base.np.array([pairs.index(tuple(sorted((types[u],types[v])))) for u,v in edges.T])
        with torch.no_grad():
            model.structure_pair_projection.weight.copy_(torch.arange(1,65).float()[:,None])
        ctx = FrozenTypedContext(model, types, edges, families, 'nonzero')
        queries = torch.tensor([(0,6),(0,1),(1,4),(2,5),(3,6),(4,6)]).T
        flags = torch.tensor([1.,0.,1.,0.,1.,0.])
        expected = flags[:,None] * torch.arange(1,65).float()[None,:]
        before = base.stable(rng_capture(rng,'cpu'))
        try:
            self.assertTrue(torch.equal(model.structure_pair_residual(ctx,queries[:,:2]),expected[:2]))
            self.assertTrue(torch.equal(model.structure_pair_residual(ctx,queries),expected))
            order = torch.tensor([5,2,0,4,1,3])
            self.assertTrue(torch.equal(model.structure_pair_residual(ctx,queries[:,order]),expected[order]))
            chunks = [model.structure_pair_residual(ctx,queries[:,s]) for s in (slice(0,1),slice(1,4),slice(4,6))]
            self.assertTrue(torch.equal(torch.cat(chunks),expected))
            self.assertEqual(before,base.stable(rng_capture(rng,'cpu')))
        finally:
            ctx.close()

    def test_reject_same_shape_schema_reordering_and_prior_change(self):
        case, cfg, ordinary, _, rng = setup()
        before = base.stable(rng_capture(rng,'cpu'))
        pairs = ordinary.family_pairs
        ordinary.family_pairs = tuple(reversed(pairs))
        with self.assertRaisesRegex(ValueError,'ordered schema'):
            StructureInputA.from_initial_A(ordinary,case,cfg,completed_updates=0)
        ordinary.family_pairs = pairs
        with torch.no_grad(): ordinary.prior[0].add_(.01)
        with self.assertRaisesRegex(ValueError,'prior/support'):
            StructureInputA.from_initial_A(ordinary,case,cfg,completed_updates=0)
        self.assertEqual(before,base.stable(rng_capture(rng,'cpu')))

    def test_save_restore_next_update_and_preforward_rollback(self):
        case, cfg, _, model, rng = setup()
        opt = torch.optim.AdamW(model.parameters(), **{k:v for k,v in cfg['optimizer'].items() if k!='type'})
        ema = ParameterEMA(model)
        tx = TrainingBoundary(model, opt, ema, rng, 'cpu', 0)
        def step(position):
            with tx.step(position):
                inp = update.one_input(case, cfg, position, rng)
                update.update(model, opt, case, inp, 1000, 'cpu', ema, tx.completed, transaction=tx)
            return base.input_hash([inp])
        def snapshot():
            p = tx.checkpoint({})
            return {k:p[k] for k in ('model','optimizer','ema','rng','updates')}
        step(1)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'state.pt'
            torch.save(tx.checkpoint({}), path)
            inp1 = step(2); expected = base.stable(snapshot())
            p = torch.load(path, map_location='cpu')
        completed = base.restore_payload(p, model, opt, ema, {}, lambda s:rng_restore(s,rng,'cpu'))
        tx = TrainingBoundary(model, opt, ema, rng, 'cpu', completed)
        before = base.stable(snapshot())
        with self.assertRaises(RuntimeError):
            with tx.step(2):
                update.one_input(case, cfg, 2, rng)
                raise RuntimeError('after input sampling; before model forward')
        self.assertTrue(tx.resumable)
        self.assertEqual(before, base.stable(snapshot()))
        self.assertEqual(inp1, step(2))
        self.assertEqual(expected, base.stable(snapshot()))

    def test_reject_trained_or_different_architecture_transfer(self):
        case, cfg, ordinary, _, rng = setup()
        with self.assertRaises(ValueError):
            StructureInputA.from_initial_A(ordinary, case, cfg, completed_updates=1)
        ordinary.arm = 'E'
        with self.assertRaises(ValueError):
            StructureInputA.from_initial_A(ordinary, case, cfg, completed_updates=0)


if __name__ == '__main__':
    unittest.main()
