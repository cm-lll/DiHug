"""CPU recovery checks with actual O/A/D/E classes and full-sized parameters.

Only the graph (12 nodes) and frozen E dictionary are synthetic engineering
fixtures. No formal data, trained weights, dictionary fitting or GPU is used.
"""
import copy
import gc
import json
import pathlib
import subprocess
import tempfile
import unittest
from unittest import mock

import test_abort_boundaries as base
import numpy as np
import torch
import integration
from boundary_transactions import TrainingBoundary, rng_capture, rng_restore
from ema import ParameterEMA, cpu_copy
from hsgd.computational import FrozenTypedContext
from model_variants import StructuralDenoiser
from probe import equal, input_hash

PARENT = '4b6764174d18b4713c2836e5524f78eb5b7895b2'
EVIDENCE = dict(reproductions=[], real_layouts={})


def reviewed(relative, name):
    path=base.TEMP/'reviewed_4b676417'/relative
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(subprocess.check_output(['git','show',PARENT+':'+relative],cwd=base.ROOT))
    return base.load(path,name)


def fixtures():
    types=np.repeat(np.arange(4),3)
    schema=[(a,b) for a in range(4) for b in range(a,4)]
    edges=np.array([(3*a,3*a+1 if a==b else 3*b) for a,b in schema],dtype=np.int64).T
    case=base.common.make_case(types,edges,tuple(schema))
    cfg=json.loads((base.SCI/'package/config.json').read_text())
    cfg.update(seed=912,inputs_per_update=1,rounds=1)
    # Full actual 8-layer / 256 / 64 / 128 architecture and PubMed role widths.
    K=[8,8,8,6];H=np.random.default_rng(903).uniform(.1,1.,(sum(K),630))
    H/=np.linalg.norm(H,axis=1,keepdims=True)
    basis=dict(basis=H,offsets=np.r_[0,np.cumsum(K)])
    return case,cfg,basis


def objects(arm):
    rng=base.reset(4301);case,cfg,basis=fixtures()
    model=StructuralDenoiser(case,cfg,arm,basis if arm=='E' else None)
    opt=torch.optim.AdamW(model.parameters(),**{k:v for k,v in cfg['optimizer'].items() if k!='type'})
    ema=ParameterEMA(model)
    EVIDENCE['real_layouts'][arm]=dict(parameters=sum(p.numel() for p in model.parameters()),
        layers=len(model.backbone.tf_layers),joint_head_input=model.joint_head[0].in_features,
        has_role_method=hasattr(model,'_role_identity'),
        role_buffers=[k for k in model._buffers if k.startswith('role_')],
        graph_nodes=12,legal_slots=case['space'].total,synthetic_dictionary=arm=='E')
    return case,cfg,model,opt,ema,rng


def state(tx):
    p=tx.checkpoint(dict(engineering_only=True))
    return base.stable({k:p[k] for k in ('model','optimizer','ema','rng','updates')})


class ReproduceReviewedBugs(unittest.TestCase):
    def test_reviewed_O_A_D_role_method_without_buffers_misclassifies_rollback(self):
        old=reviewed(str((base.SCI/'boundary_transactions.py').relative_to(base.ROOT)), 'reviewed_old_boundary')
        update=base.load(base.ROOT/'experiments/o_cn_matched_execute_v1/single_input.py','repro_input')
        for arm in ('O','A','D','E'):
            case,cfg,model,opt,ema,rng=objects(arm)
            tx=old.TrainingBoundary(model,opt,ema,rng,'cpu',0)
            with self.assertRaises(ValueError):
                with tx.step(1):
                    update.one_input(case,cfg,1,rng)
                    raise ValueError('injected after real input sampling')
            entry=dict(bug='role_method_without_buffers',arm=arm,resumable=tx.resumable,failure=tx.last_failure)
            EVIDENCE['reproductions'].append(entry)
            self.assertEqual(tx.resumable,arm=='E')
            if arm!='E':self.assertIn('role_basis',tx.last_failure['rollback_error'])
            del tx,model,opt,ema;gc.collect()

    def test_reviewed_partial_raw_restore_can_be_checkpointed_as_exact(self):
        old=reviewed(str((base.SCI/'integration.py').relative_to(base.ROOT)), 'reviewed_old_integration')
        case,cfg,model,opt,ema,rng=objects('O')
        # Distinct readout parameters are an engineering fixture, not a new EMA rule.
        with torch.no_grad():
            for v in ema.shadow.values():v.add_(.125)
        tx=TrainingBoundary(model,opt,ema,rng,'cpu',0)
        raw=cpu_copy(model.state_dict());load=model.load_state_dict;calls=[0]
        def partial(sd,strict=True):
            calls[0]+=1
            if calls[0]==2:
                key='joint_head.2.bias'
                torch.nn.Module.load_state_dict(model,{key:sd[key]},strict=False)
                raise RuntimeError('injected partial raw restore')
            return load(sd,strict=strict)
        with mock.patch.object(model,'load_state_dict',partial),self.assertRaises(RuntimeError):
            old.ema_validation(lambda *a:None,model,ema)
        stopped=tx.checkpoint(dict(engineering_only=True))
        self.assertFalse(equal(raw,stopped['model']))
        self.assertTrue(stopped['recovery']['exact_resume_allowed'])
        EVIDENCE['reproductions'].append(dict(bug='partial_raw_restore',mixed_state=True,
            checkpoint_marked_exact=stopped['recovery']['exact_resume_allowed']))


class ActualLayoutRollback(unittest.TestCase):
    def path(self,arm,fault=None):
        case,cfg,model,opt,ema,rng=objects(arm)
        name='o_cn_matched_execute_v1' if arm in ('O','A') else 'late_final_single_input_v1'
        update=base.load(base.ROOT/'experiments'/name/'single_input.py','real_update_'+arm)
        tx=TrainingBoundary(model,opt,ema,rng,'cpu',0);trace=[]
        for u in (1,2):
            if u==2 and fault is not None:
                before=state(tx);hook=None
                with self.assertRaises((ValueError,RuntimeError,FloatingPointError)):
                    with tx.step(u):
                        inp=update.one_input(case,cfg,u,rng)
                        if fault=='input':raise ValueError('real input sampled')
                        if fault=='forward':
                            def fail(*a):raise RuntimeError('real model head forward failure')
                            hook=model.joint_head.register_forward_pre_hook(fail)
                        if fault=='gradient':
                            hook=model.joint_head[-1].bias.register_hook(lambda g:g*float('nan'))
                        try:update.update(model,opt,case,inp,1000,'cpu',ema,tx.completed,transaction=tx)
                        finally:
                            if hook:hook.remove()
                self.assertTrue(tx.resumable,tx.last_failure)
                self.assertEqual(before,state(tx));self.assertEqual(tx.completed,1)
                with tempfile.TemporaryDirectory(dir=base.TEMP) as d:
                    path=pathlib.Path(d)/'complete.pt';torch.save(tx.checkpoint({}),path);p=torch.load(path,map_location='cpu')
                del tx,model,opt,ema;gc.collect()
                _,_,model,opt,ema,rng=objects(arm)
                completed=integration.restore_payload(p,model,opt,ema,{},lambda s:rng_restore(s,rng,'cpu'))
                tx=TrainingBoundary(model,opt,ema,rng,'cpu',completed);del p
            with tx.step(u):
                inp=update.one_input(case,cfg,u,rng)
                update.update(model,opt,case,inp,1000,'cpu',ema,tx.completed,transaction=tx)
            trace.append((input_hash([inp]),state(tx)))
        del tx,model,opt,ema;gc.collect()
        return trace

    def test_real_O_A_D_E_input_forward_gradient_failures_resume_identically(self):
        for arm in ('O','A','D','E'):
            with self.subTest(arm=arm):
                expected=self.path(arm)
                for fault in ('input','forward','gradient'):
                    with self.subTest(fault=fault):self.assertEqual(expected,self.path(arm,fault))


class ValidationBoundaries(unittest.TestCase):
    def trained(self,arm):
        case,cfg,model,opt,ema,rng=objects(arm)
        update=base.load(base.ROOT/'experiments/late_final_single_input_v1/single_input.py','validation_update')
        tx=TrainingBoundary(model,opt,ema,rng,'cpu',0)
        with tx.step(1):
            inp=update.one_input(case,cfg,1,rng)
            update.update(model,opt,case,inp,1000,'cpu',ema,0,transaction=tx)
        return case,cfg,model,opt,ema,rng,tx,update

    def probe(self,model,case,cfg,fail=False):
        # Real current-Xt model evaluation, independent fixed noise input.
        private=np.random.default_rng(937)
        inp=base.common.train_input(case,1000,1,cfg['seed'],9001,private,force_t=500)
        old_mode=model.training;model.eval()
        ctx=FrozenTypedContext(model,case['space'].types,inp['edges'],inp['edge_families'],'validation-test')
        try:
            with torch.no_grad():_,_,p=ctx.predict(inp['query'],inp['f'],.5,'validation-test',details=True)
            # All three global streams must be restored, even after an error.
            base.random.random();np.random.random();torch.rand(3)
            if fail:raise ValueError('injected ordinary validation failure')
            return float(p.sum())
        finally:ctx.close();model.train(old_mode)

    def continuation(self,arm,variant):
        case,cfg,model,opt,ema,rng,tx,update=self.trained(arm)
        before=state(tx)
        if variant!='none':
            try:
                with tx.validation():
                    # Both boundary and direct serializer must refuse transient weights.
                    with self.assertRaises(RuntimeError):tx.checkpoint({})
                    with self.assertRaises(RuntimeError):integration.checkpoint_payload(model,opt,ema,1,rng_capture(rng,'cpu'),{})
                    self.probe(model,case,cfg,fail=variant=='raw_error')
                    integration.ema_validation(lambda m:self.probe(m,case,cfg,fail=variant=='ema_error'),model,ema,boundary=tx)
            except ValueError:
                self.assertIn(variant,('raw_error','ema_error'))
            else:self.assertEqual(variant,'normal')
            self.assertTrue(tx.resumable,tx.last_failure)
            self.assertEqual(before,state(tx))
            self.assertTrue(equal(tx.boundary_rng,rng_capture(rng,'cpu')))
        with tx.step(2):
            inp=update.one_input(case,cfg,2,rng)
            update.update(model,opt,case,inp,1000,'cpu',ema,1,transaction=tx)
        answer=(input_hash([inp]),state(tx))
        del tx,model,opt,ema;gc.collect()
        return answer

    def test_real_models_normal_and_failed_validation_with_verified_restore(self):
        for arm in ('O','A','D','E'):
            with self.subTest(arm=arm):
                expected=self.continuation(arm,'none')
                for variant in ('normal','raw_error','ema_error'):
                    with self.subTest(variant=variant):self.assertEqual(expected,self.continuation(arm,variant))

    def test_partial_raw_restore_is_poisoned_and_checkpoint_is_forbidden(self):
        for arm in ('O','E'):
            with self.subTest(arm=arm):
                case,cfg,model,opt,ema,rng,tx,update=self.trained(arm)
                previous=tx.checkpoint({});raw=cpu_copy(model.state_dict());readout=ema.readout_state_dict(model)
                differing=[k for k,v in raw.items() if not torch.equal(v,readout[k])]
                self.assertGreater(len(differing),1)
                original=model.load_state_dict;calls=[0]
                def broken(sd,strict=True):
                    calls[0]+=1
                    if calls[0]==2:
                        torch.nn.Module.load_state_dict(model,{differing[0]:sd[differing[0]]},strict=False)
                        raise RuntimeError('partial raw restore failed')
                    return original(sd,strict=strict)
                with mock.patch.object(model,'load_state_dict',broken),self.assertRaises(RuntimeError):
                    with tx.validation():
                        integration.ema_validation(lambda m:self.probe(m,case,cfg),model,ema,boundary=tx)
                self.assertFalse(tx.resumable);self.assertEqual(tx.phase,'invalid')
                self.assertFalse(ema.at_update_boundary)
                self.assertFalse(tx.last_failure['exact_resume_allowed'])
                with self.assertRaises(RuntimeError):tx.checkpoint({})
                with self.assertRaises(RuntimeError):integration.checkpoint_payload(model,opt,ema,1,rng_capture(rng,'cpu'),{})
                # The independent PREVIOUS complete checkpoint is still valid in fresh objects.
                del tx,model,opt,ema;gc.collect()
                _,_,model,opt,ema,rng=objects(arm)
                u=integration.restore_payload(previous,model,opt,ema,{},lambda s:rng_restore(s,rng,'cpu'))
                tx=TrainingBoundary(model,opt,ema,rng,'cpu',u)
                self.assertTrue(equal(previous,tx.checkpoint({})))
                del tx,model,opt,ema;gc.collect()

    def test_direct_ema_helper_also_refuses_mixed_raw_state_without_owner(self):
        case,cfg,model,opt,ema,rng,tx,update=self.trained('A')
        original=model.load_state_dict;calls=[0]
        def broken(sd,strict=True):
            calls[0]+=1
            if calls[0]==2:
                key=next(iter(dict(model.named_parameters())))
                torch.nn.Module.load_state_dict(model,{key:sd[key]},strict=False)
                raise RuntimeError('standalone partial raw restore')
            return original(sd,strict=strict)
        with mock.patch.object(model,'load_state_dict',broken),self.assertRaises(RuntimeError):
            integration.ema_validation(lambda m:None,model,ema)
        with self.assertRaises(RuntimeError):tx.checkpoint({})
        with self.assertRaises(RuntimeError):integration.checkpoint_payload(model,opt,ema,1,rng_capture(rng,'cpu'),{})

    def test_unreported_restore_mutation_or_optimizer_mutation_is_not_certified(self):
        for kind in ('raw','optimizer','role_cache'):
            case,cfg,model,opt,ema,rng,tx,update=self.trained('E')
            with self.assertRaises(RuntimeError):
                with tx.validation():
                    if kind=='raw':
                        with torch.no_grad():next(model.parameters()).add_(1.)
                    elif kind=='optimizer':next(iter(opt.state.values()))['exp_avg'].add_(1.)
                    else:model._basis_hash='invalid hidden role cache'
            self.assertFalse(tx.resumable)
            with self.assertRaises(RuntimeError):tx.checkpoint({})
            del tx,model,opt,ema;gc.collect()

    def test_all_formal_validation_helpers_are_inside_boundary(self):
        import ast
        for name in ('o_cn_matched_execute_v1/worker_run.py','late_final_single_input_v1/worker_run.py','d_late_ard_runtime_v1/generation_worker.py'):
            tree=ast.parse((base.ROOT/'experiments'/name).read_text());functions=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='validate']
            self.assertEqual(len(functions),1)
            protected=[n for n in ast.walk(functions[0]) if isinstance(n,ast.With) and any(isinstance(x.context_expr,ast.Call) and isinstance(x.context_expr.func,ast.Attribute) and x.context_expr.func.attr=='validation' for x in n.items)]
            self.assertEqual(len(protected),1)
            calls=[n for n in ast.walk(protected[0]) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='ema_validation']
            self.assertEqual(len(calls),1);self.assertTrue(any(k.arg=='boundary' for k in calls[0].keywords))


if __name__=='__main__':unittest.main(verbosity=2)
