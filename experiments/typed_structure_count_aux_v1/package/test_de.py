"""Combination-only CPU checks. No optimizer updates or sampled graph."""
import unittest
from unittest.mock import patch
import numpy as np
import torch
from test_oabc import fixture, prediction
from test_engineering import config
from model_variants import StructuralDenoiser, copy_common
from normal2 import normal2, pair_inputs
from hsgd.computational import FrozenTypedContext
from hsgd.bw_preflight.common import plan_for, make_case

torch.set_num_threads(1)

class Combination(unittest.TestCase):
    def models(self):
        c,b=fixture();torch.manual_seed(411)
        a=StructuralDenoiser(c,config(),'A')
        models={}
        for arm in 'DE':
            m=StructuralDenoiser(c,config(),arm,b if arm=='E' else None)
            copy_common(a,m);models[arm]=m
        return c,a,models

    def test_paths_and_zero_initial_baseline(self):
        c,a,models=self.models();_,(_,la,pa)=prediction(c,a)
        for arm,m in models.items():
            self.assertIsNotNone(m.common_neighbor)
            self.assertEqual(m.joint_head[0].in_features,a.joint_head[0].in_features)
            self.assertTrue(hasattr(m,'refex_projection'))
            self.assertEqual(hasattr(m,'role_projection'),arm=='E')
            if arm=='D':
                with patch('model_variants.encode_typed',side_effect=AssertionError('D used roles')):
                    _,(_,l,p)=prediction(c,m)
            else:_,(_,l,p)=prediction(c,m)
            torch.testing.assert_close(la,l,atol=0,rtol=0)
            torch.testing.assert_close(pa,p,atol=0,rtol=0)

    def test_joint_nonzero_features_and_gradient_paths(self):
        c,a,models=self.models()
        for arm,m in models.items():
            with torch.no_grad():
                m.refex_projection.weight.fill_(.01)
                if arm=='E':m.role_projection.weight.fill_(.02)
            _,(_,l,p)=prediction(c,m);_,(_,la,pa)=prediction(c,a)
            self.assertGreater(float((l-la).abs().max()),1e-7)
            l.square().sum().backward()
            for name in ['refex_projection.weight','common_neighbor.phi.0.weight']+(['role_projection.weight'] if arm=='E' else []):
                grad=dict(m.named_parameters())[name].grad
                self.assertIsNotNone(grad);self.assertTrue(torch.isfinite(grad).all())
                self.assertGreater(float(grad.abs().max()),0)
            if arm=='E':self.assertIsNone(m.role_basis.grad)

    def test_query_edges_do_not_enter_Xt_CN(self):
        c,_,models=self.models();ids,f,q=plan_for(c,13,1).block(0)
        for m in models.values():
            ctx=FrozenTypedContext(m,c['space'].types,c['clean'],c['space'].encode(c['clean'])[0],1)
            old=ctx.old.copy();keys=ctx.cn_wedges.keys.clone()
            ctx.predict(q,f,.5,1)
            np.testing.assert_array_equal(old,ctx.old)
            self.assertTrue(torch.equal(keys,ctx.cn_wedges.keys))
            self.assertGreater(q.shape[1],old.shape[1]);ctx.close()

    def test_nonzero_node_permutation(self):
        c,_,models=self.models();perm=np.array([2,0,1,5,3,4])
        types=np.empty_like(c['space'].types);types[perm]=c['space'].types
        other=make_case(types,np.sort(perm[c['clean']],axis=0),c['space'].pairs)
        for m in models.values():
            with torch.no_grad():
                m.refex_projection.weight.fill_(.01)
                if m.arm=='E':m.role_projection.weight.fill_(.02)
            ids,(_,l,p)=prediction(c,m);ids2,(_,l2,p2)=prediction(other,m)
            _,_,q=plan_for(c,13,1).block(0)
            f,local=other['space'].encode(np.sort(perm[q],axis=0));mapped=other['space'].offsets[f]+local
            lookup={int(i):j for j,i in enumerate(ids2)};order=[lookup[int(i)] for i in mapped]
            torch.testing.assert_close(l,l2[order],atol=2e-6,rtol=1e-5)
            torch.testing.assert_close(p,p2[order],atol=2e-6,rtol=1e-5)

    def test_endpoint_symmetry_and_edge_order(self):
        c,_,models=self.models();_,f,q=plan_for(c,13,1).block(0)
        for m in models.values():
            with torch.no_grad():
                m.refex_projection.weight.fill_(.01)
                if m.arm=='E':m.role_projection.weight.fill_(.02)
            _,(_,l,p)=prediction(c,m)
            _,(_,l2,p2)=prediction(c,m,np.sort(c['clean'][::-1,::-1],axis=0))
            torch.testing.assert_close(l,l2,atol=2e-6,rtol=1e-5)
            ctx=FrozenTypedContext(m,c['space'].types,c['clean'],c['space'].encode(c['clean'])[0],1)
            tq=torch.as_tensor(q);tf=torch.as_tensor(f)
            ctx.node_residual=m.branch(m.branch.input(ctx,.5))
            z=m.common_neighbor.aggregate(ctx,tq,tf,.5)
            zr=m.common_neighbor.aggregate(ctx,tq.flip(0),tf,.5)
            torch.testing.assert_close(z,zr,atol=2e-6,rtol=1e-5);ctx.close()
            # Public graph API deliberately requires canonical endpoint order.

    def test_cache_is_frozen_context_only(self):
        c,_,models=self.models();_,f,q=plan_for(c,13,1).block(0)
        for m in models.values():
            ctx=FrozenTypedContext(m,c['space'].types,c['clean'],c['space'].encode(c['clean'])[0],1)
            with torch.no_grad():ctx.predict(q,f,.5,1);ctx.predict(q,f,.5,1)
            self.assertEqual(m.feature_calls,1)
            with self.assertRaises(ValueError):ctx.predict(q,f,.5,2)
            with torch.no_grad():m.refex_projection.weight.add_(.001)
            with self.assertRaises(ValueError):ctx.predict(q,f,.5,1)
            ctx.close();prediction(c,m);self.assertEqual(m.feature_calls,2)

    def test_normal2_optimizer_membership_no_step(self):
        c,_,models=self.models()
        for m in models.values():
            opt=torch.optim.AdamW(m.parameters())
            self.assertEqual({id(p) for g in opt.param_groups for p in g['params']},{id(p) for p in m.parameters()})
            inp=pair_inputs(c,dict(T=1000,seed=9),1,np.random.default_rng(11),500)
            with patch.object(opt,'step',side_effect=AssertionError('CPU test cannot update')):
                rec=normal2(m,opt,c,inp,1000,'cpu',False)
            self.assertEqual(rec['optimizer_steps'],0)
            self.assertTrue(torch.isfinite(m.refex_projection.weight.grad).all())

    def test_E_dictionary_reload_exact(self):
        c,_,models=self.models();m=models['E'];_,b=fixture()
        n=StructuralDenoiser(c,config(),'E',b);n.load_state_dict(m.state_dict(),strict=True)
        for k,v in m.state_dict().items():self.assertTrue(torch.equal(v,n.state_dict()[k]))
        torch.testing.assert_close(prediction(c,m)[1][1],prediction(c,n)[1][1],atol=0,rtol=0)

if __name__=='__main__':unittest.main()
