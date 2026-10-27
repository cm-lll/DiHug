import copy,json,unittest,sys
import numpy as np
import torch
from features import extract,encode,fit_basis,SPEC,feature_width
from model_variants import StructuralDenoiser,copy_common
from prepare import load_case,ROOT
from hsgd.bw_preflight.common import make_case,plan_for
from hsgd.computational import FrozenTypedContext
from hsgd.model import TypedSparseDenoiser
from hsgd.losses import supported_binary_ce

torch.set_num_threads(1)

def toy():
    return make_case(np.array([0,0,0,1,1,1]),np.array([[0,0,1,1,2,3],[1,3,2,4,5,4]]),((0,0),(0,1),(1,1)))

def config():
    # tiny engineering tests only; real probes use unmodified 8-layer config.
    return {'model':dict(layers=2,width=16,edge_width=8,global_width=8,heads=2,ff_node=16,ff_edge=8,ff_global=8,dropout=.1,cn_hidden=32,joint_width=16)}

def model(arm):
    c=toy();f=extract(c['space'].types,c['clean'],c['space'].encode(c['clean'])[0],c['space'].pairs)
    rng=np.random.default_rng(41);basis=rng.uniform(.01,1,(2,4,f.shape[1]));basis/=np.linalg.norm(basis,axis=2)[:,:,None]
    return c,StructuralDenoiser(c,config(),arm,basis)

def context(c,m,token=1):return FrozenTypedContext(m,c['space'].types,c['clean'],c['space'].encode(c['clean'])[0],token)
def query(c):return plan_for(c,42,2).block(0)[1:]
def pred(c,m,ctx=None):
    f,q=query(c);own=ctx is None;ctx=context(c,m) if own else ctx
    try:return ctx.predict(q,f,.5,1,details=True)
    finally:
        if own:ctx.close()

class Features(unittest.TestCase):
    def test_independent_base_counts(self):
        c=toy();ty=c['space'].types;e=c['clean'];f=c['space'].encode(e)[0];x=extract(ty,e,f,c['space'].pairs)
        for node in range(len(ty)):
            neighbors={int(v) for u,v in e.T if u==node}|{int(u) for u,v in e.T if v==node};eg=neighbors|{node}
            deg=np.array([sum(node in (u,v) and ff==r for (u,v),ff in zip(e.T,f)) for r in range(3)])
            ins=np.array([sum(u in eg and v in eg and ff==r for (u,v),ff in zip(e.T,f)) for r in range(3)])
            out=np.array([sum(((u in eg)!=(v in eg)) and ff==r for (u,v),ff in zip(e.T,f)) for r in range(3)])
            base=np.r_[np.log1p(deg)/np.log1p(5),np.log1p(ins)/np.log1p(c['space'].counts),np.log1p(out)/np.log1p(c['space'].counts)]
            np.testing.assert_allclose(x[node,:9],base,atol=1e-10,rtol=0)
    def test_empty(self):
        c=toy();x=extract(c['space'].types,np.empty((2,0),int),np.array([],int),c['space'].pairs)
        self.assertEqual(x.shape,(6,feature_width(3)));self.assertTrue((x==0).all())
    def test_permutation(self):
        c=toy();p=np.array([2,0,1,5,3,4]);ty=c['space'].types;e=c['clean'];f=c['space'].encode(e)[0]
        ty2=np.empty_like(ty);ty2[p]=ty;e2=np.sort(p[e],axis=0)
        a=extract(ty,e,f,c['space'].pairs);b=extract(ty2,e2,f,c['space'].pairs)
        np.testing.assert_allclose(a,b[p],atol=1e-10,rtol=0)
    def test_edge_order(self):
        c=toy();e=c['clean'];f=c['space'].encode(e)[0];a=extract(c['space'].types,e,f,c['space'].pairs)
        b=extract(c['space'].types,e[:,::-1],f[::-1],c['space'].pairs);np.testing.assert_array_equal(a,b)
    def test_reject(self):
        c=toy();e=c['clean'];f=c['space'].encode(e)[0]
        for ee,ff in [(e[::-1],f),(np.c_[e,e[:,0]],np.r_[f,f[0]]),(e,1-f)]:
            with self.assertRaises(ValueError):extract(c['space'].types,ee,ff,c['space'].pairs)
    def test_nnls_strength_and_permutation(self):
        f=np.eye(4);basis=np.eye(4);a,_=encode(f,basis);b,_=encode(3*f,basis)
        np.testing.assert_allclose(b,3*a);self.assertTrue((b.sum(1)==3).all())
        np.testing.assert_allclose(encode(f[::-1],basis)[0],a[::-1])

class Integration(unittest.TestCase):
    def test_A_original_exact(self):
        sys.path.insert(0,str(ROOT/'audit_reference'))
        from pair_model import PairDenoiser
        c,a=model('A');old=PairDenoiser(c['n_types'],c['n_families'],family_pairs=c['space'].pairs,prior=c['prior'],T=1000,arm='base',**config()['model'])
        with torch.no_grad():a.branch.degree_out.weight.normal_(0,.02)
        state=old.state_dict()
        for k,v in a.state_dict().items():state[k]=v.clone()
        old.load_state_dict(state);old.set_target(None)
        la=pred(c,a)[1];lb=pred(c,old)[1]
        torch.testing.assert_close(la,lb,atol=0,rtol=0)
        la.square().sum().backward();lb.square().sum().backward()
        other=dict(old.named_parameters())
        for name,param in a.named_parameters():
            if param.grad is None:self.assertIsNone(other[name].grad)
            else:torch.testing.assert_close(param.grad,other[name].grad,atol=0,rtol=0)
    def test_B_C_initial_equal_and_no_CN(self):
        c,b=model('B');_,d=model('C');copy_common(b,d)
        self.assertIsNone(b.common_neighbor);self.assertIsNone(d.common_neighbor)
        self.assertEqual(d.cn_feature_width,0);self.assertFalse(any('common_neighbor' in k for k in d.state_dict()))
        torch.testing.assert_close(pred(c,b)[1],pred(c,d)[1],atol=0,rtol=0)
        ctx=context(c,d);self.assertIsNone(ctx.cn_wedges);ctx.close()
    def test_CN_constructor_unreachable_in_B_C(self):
        from unittest.mock import patch
        with patch('hsgd.common_neighbor.Wedges',side_effect=AssertionError('CN path reached')):
            for arm in ['B','C']:
                c,m=model(arm);pred(c,m)
    def test_nonzero_projection_response_gradient(self):
        c,m=model('C');base=pred(c,m)[1].detach();
        with torch.no_grad():m.role_projection.weight.normal_(0,.02)
        result=pred(c,m);self.assertGreater(float((result[1]-base).abs().max()),1e-6)
        result[1].square().sum().backward()
        for name in ['refex_projection','role_projection']:
            g=getattr(m,name).weight.grad;self.assertTrue(torch.isfinite(g).all());self.assertGreater(float(g.abs().max()),0)
    def test_query_exclusion_and_reuse(self):
        c,m=model('C');ctx=context(c,m);pred(c,m,ctx);before=ctx.structural_features.clone()
        f,q=plan_for(c,77,2).block(1)[1:];ctx.predict(q,f,.5,1)
        self.assertEqual(m.feature_calls,1);torch.testing.assert_close(before,ctx.structural_features,atol=0,rtol=0)
        independent=extract(ctx.types,ctx.old,ctx.of,ctx.pairs);np.testing.assert_allclose(before.numpy(),independent,atol=1e-7)
        ctx.close()
        with self.assertRaises(ValueError):ctx.predict(q,f,.5,1)
    def test_cache_time_update_basis_guards(self):
        for what in ['time','param','basis']:
            c,m=model('C');ctx=context(c,m);pred(c,m,ctx);f,q=query(c)
            with torch.no_grad():
                if what=='param':m.refex_projection.weight.add_(.01)
                if what=='basis':m.role_basis.add_(.01)
            with self.assertRaises(ValueError):ctx.predict(q,f,.6 if what=='time' else .5,1)
            ctx.close()
    def test_basis_reload_and_new_state(self):
        c,m=model('C');pred(c,m);state=copy.deepcopy(m.state_dict());state['role_basis']=state['role_basis']*.9
        m.load_state_dict(state);pred(c,m)
        changed=copy.deepcopy(c);changed['clean']=c['clean'][:,:-1]
        ctx=context(changed,m);pred(changed,m,ctx);self.assertEqual(ctx.old.shape[1],c['clean'].shape[1]-1);ctx.close()
    def test_nonzero_full_equivariance(self):
        c,m=model('C')
        with torch.no_grad():m.refex_projection.weight.normal_(0,.01);m.role_projection.weight.normal_(0,.01)
        p=np.array([2,0,1,5,3,4]);f,q=query(c);e=c['clean'];ty=np.empty_like(c['space'].types);ty[p]=c['space'].types
        with torch.no_grad():
            a=m(c['space'].types,e,c['space'].encode(e)[0],q,f,.5)
            b=m(ty,np.sort(p[e],axis=0),c['space'].encode(e)[0],np.sort(p[q],axis=0),f,.5)
        torch.testing.assert_close(a,b,atol=2e-6,rtol=1e-5)
    def test_train_eval_same_input(self):
        c,m=model('C');m.train();a=pred(c,m)[1];m.eval();b=pred(c,m)[1]
        torch.testing.assert_close(a,b,atol=0,rtol=0)
    def test_support_and_ce(self):
        c,m=model('B');graph,logits,p=pred(c,m);labels=torch.zeros(len(p));prior=m.prior[graph.query_families]
        loss=supported_binary_ce(logits,labels,prior).sum();self.assertTrue(torch.isfinite(loss))
    def test_received_datasets(self):
        for name in ['IMDB','PubMed']:
            c,cfg=load_case(name);self.assertEqual(len(c['space'].types),192)
            self.assertEqual(cfg['rounds'],1);self.assertEqual(cfg['inputs_per_update'],2)
            plan=plan_for(c,45,1);seen=[]
            for r in range(1):seen.extend(plan.block(r)[0].tolist())
            self.assertEqual(len(seen),len(set(seen)));self.assertEqual(len(seen),int(c['space'].counts[c['active']].sum()))

class Revision(unittest.TestCase):
    def test_relation_specific_one_round_independent(self):
        c=toy();ty=c['space'].types;e=c['clean'];fam=c['space'].encode(e)[0];x=extract(ty,e,fam,c['space'].pairs);base=x[:,:9]
        for r in range(3):
            for i in range(6):
                nb=[int(v if u==i else u) for (u,v),f in zip(e.T,fam) if f==r and i in (u,v)]
                sums=base[nb].sum(0) if nb else np.zeros(9);mean=sums/len(nb) if nb else sums
                np.testing.assert_allclose(x[i,9+18*r:9+18*(r+1)],np.r_[mean,sums/5],atol=1e-10)
    def test_typed_coefficients_layout_strength(self):
        types=np.array([0,1,0,1]);f=3*np.eye(4);basis=np.stack([np.eye(4),2*np.eye(4)])
        co,_=encode(f,basis,types)
        for i,t in enumerate(types):
            np.testing.assert_array_equal(co[i,4*(1-t):4*(2-t)],0)
            np.testing.assert_allclose(co[i,4*t:4*t+4],f[i]/(1+t))
        p=np.array([2,0,3,1]);np.testing.assert_allclose(encode(f[p],basis,types[p])[0],co[p])
    def test_normal2_sequential_gradient_equivalence(self):
        from normal2 import normal2,pair_inputs
        from hsgd.bw_preflight.common import update
        c,m=model('C');cfg=dict(T=1000,seed=41);inputs=pair_inputs(c,cfg,0,np.random.default_rng(51),500)
        _,m2=model('C');m2.load_state_dict(copy.deepcopy(m.state_dict()));o=torch.optim.AdamW(m.parameters());o2=torch.optim.AdamW(m2.parameters())
        diag={};normal2(m,o,c,inputs,1000,'cpu',step=False,diagnostics=diag)
        ds=[]
        for inp in inputs:
            d={};update(m2,o2,c,inp,1000,'cpu',do_step=False,diagnostics=d);ds.append(d)
        for k,g in diag['gradients'].items():
            if g is None:self.assertIsNone(ds[0]['gradients'][k])
            else:torch.testing.assert_close(g,(ds[0]['gradients'][k]+ds[1]['gradients'][k])*.5,atol=2e-6,rtol=1e-5)
        self.assertEqual(m.feature_calls,2)
    def test_normal2_zero_step_order_and_context_release(self):
        from normal2 import normal2,pair_inputs
        from unittest.mock import patch
        c,m=model('C');inp=pair_inputs(c,dict(T=1000,seed=41),0,np.random.default_rng(8),500)
        o=torch.optim.AdamW(m.parameters());events=[];zero=o.zero_grad;step=o.step
        # Fake optimizer step avoids spending an actual engineering update.
        def z(*a,**k):events.append('zero');return zero(*a,**k)
        def s(*a,**k):events.append('step');return None
        original=FrozenTypedContext.close
        def close(ctx):events.append('close');return original(ctx)
        with patch.object(o,'zero_grad',z),patch.object(o,'step',s),patch.object(FrozenTypedContext,'close',close):
            normal2(m,o,c,inp,1000,'cpu',step=False)
        self.assertEqual(events,['zero','close','close'])
    def test_reject_partial_queries(self):
        from normal2 import normal2,pair_inputs
        c,m=model('B');inp=pair_inputs(c,dict(T=1000,seed=41),0,np.random.default_rng(8),500)
        inp[0]['ids']=inp[0]['ids'][:-1]
        with self.assertRaises(ValueError):normal2(m,torch.optim.AdamW(m.parameters()),c,inp,1000,'cpu',step=False)

if __name__=='__main__':unittest.main(verbosity=2)
