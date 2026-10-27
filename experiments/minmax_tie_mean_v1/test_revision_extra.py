"""Additional isolated protocol regressions, zero real optimizer updates."""
import copy,unittest
from unittest.mock import patch
import numpy as np
import torch
from test_engineering import model
from normal2 import normal2,pair_inputs
from hsgd.computational import FrozenTypedContext
from probe import equal,metric

class Transaction(unittest.TestCase):
    def test_one_step_after_two_backward_and_closed_contexts(self):
        c,m=model('C');inputs=pair_inputs(c,dict(T=1000,seed=41),0,np.random.default_rng(84),500)
        opt=torch.optim.AdamW(m.parameters());events=[];zero=opt.zero_grad;close=FrozenTypedContext.close
        def z(*a,**kw):events.append('zero');return zero(*a,**kw)
        def fake_step():
            events.append('step')
            with torch.no_grad():m.backbone.lin_out_E.weight.add_(.001)
        def closed(ctx):events.append('close');return close(ctx)
        handle=m.backbone.lin_out_E.weight.register_hook(lambda g:(events.append('backward'),g)[1])
        try:
            with patch.object(opt,'zero_grad',z),patch.object(opt,'step',fake_step),patch.object(FrozenTypedContext,'close',closed):
                result=normal2(m,opt,c,inputs,1000,'cpu',step=True)
        finally:handle.remove()
        self.assertEqual(events,['zero','backward','close','backward','close','step'])
        self.assertEqual(m.feature_calls,2);self.assertEqual(result['optimizer_steps'],1)
        self.assertEqual(len(opt.state),0) # no real optimization in this mock protocol check
    def test_optimizer_tree_difference_not_silently_accepted(self):
        state={'state':{0:{'step':torch.tensor(1.),'exp_avg':torch.zeros(2),'exp_avg_sq':torch.ones(2)}},'param_groups':[{'lr':.0002}]}
        altered=copy.deepcopy(state);altered['state'][0]['exp_avg'][0]=1e-9
        self.assertFalse(equal(state,altered))
        self.assertTrue(metric(state['state'][0]['exp_avg'],altered['state'][0]['exp_avg'])['passes'])
        # Continuation requires optimizer exactness in addition to FP32 tolerance.
        self.assertFalse(True and equal(state,altered))
    def test_basis_no_real_training_material_in_binding(self):
        from prepare import ROOT,load_case
        import json
        for name in ['IMDB','PubMed']:
            c,cfg=load_case(name);entry=cfg['datasets'][name]
            self.assertEqual(entry['basis_kind'],'synthetic engineering only')
            receipt=json.loads((ROOT/'assets'/(name+'_synthetic_receipt.json')).read_text())
            self.assertIn('no reference graph fit',receipt['purpose'])
            basis=np.load(ROOT/entry['basis_file'])['basis'];self.assertEqual(basis.shape,(4,4,630))

if __name__=='__main__':unittest.main()
