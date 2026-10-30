"""CPU-only checks for the newly restored query/loss contract, no model calls."""
import copy,unittest
import numpy as np
import block20 as B
from hsgd.bw_preflight.common import make_case,plan_for,query_inventory,train_input

class ProtocolTests(unittest.TestCase):
    def test_failed_preflight_exit_status(self):
        from probe_local import result_exit_code
        self.assertEqual(result_exit_code({'status':'passed'}),0)
        for status in ('failed','running',None):self.assertEqual(result_exit_code({'status':status}),1)
    def test_full_data_and_denominators(self):
        for d,total,active in [('IMDB',1365378,997313),('PubMed',1223830,1223830)]:
            c,cfg=B.load_case(d);self.assertEqual(c['space'].total,total)
            inv=query_inventory(c,plan_for(c,20261030,20));self.assertEqual(inv['active_slots'],active);self.assertEqual(inv['rounds'],20)
            self.assertEqual(inv['full_schema_ce_weight'],20/total)
            self.assertNotEqual(total,18336)
    def test_original_rng_and_input_order(self):
        c,cfg=B.load_case('IMDB');rng=np.random.default_rng(79);old=np.random.default_rng(79)
        pair=B.pair_inputs(c,cfg,3,rng,500)
        for j,inp in enumerate(pair):
            expected=train_input(c,1000,20,cfg['seed'],6+j,old,500)
            for k in ('ids','f','query','labels','edges','edge_families'):self.assertTrue(np.array_equal(inp[k],expected[k]),k)
            B.validate_input(c,inp)
        self.assertFalse(np.array_equal(pair[0]['edges'],pair[1]['edges']))
        self.assertEqual(rng.bit_generator.state,old.bit_generator.state)
    def test_bad_query_and_old_full_query_weight_rejected(self):
        c,cfg=B.load_case('IMDB');inp=B.pair_inputs(c,cfg,0,np.random.default_rng(3),500)[0]
        bad=copy.deepcopy(inp);bad['ce_weight']=1/c['space'].total
        with self.assertRaisesRegex(ValueError,'scaling'):B.validate_input(c,bad)
        bad=copy.deepcopy(inp);bad['labels'][0]=1-bad['labels'][0]
        with self.assertRaisesRegex(ValueError,'labels'):B.validate_input(c,bad)
        bad=copy.deepcopy(inp);bad['ids'][0]=bad['ids'][1]
        with self.assertRaisesRegex(ValueError,'query block'):B.validate_input(c,bad)
    def test_uniform_block_ce_and_normal2_coefficients(self):
        c=make_case(np.array([0,0,0,1,1]),np.array([[0,0],[1,3]]),((0,0),(0,1),(1,1)))
        plan=plan_for(c,76,20);loss=np.linspace(.01,.1,c['space'].total);loss[np.repeat(~c['active'],c['space'].counts)]=0
        estimate=np.mean([20*loss[plan.block(r)[0]].sum()/c['space'].total for r in range(20)])
        self.assertAlmostEqual(estimate,loss.sum()/c['space'].total,places=15)
        # Two sequential half losses; averaging does not multiply S by20.
        self.assertEqual(.5*(20*2/10)+.5*(20*3/10),5.)
    def test_roles_require_full_graph_dictionary(self):
        c,cfg=B.load_case('IMDB')
        for arm in ('C','E'):
            with self.assertRaisesRegex(RuntimeError,'dictionary unavailable'):B.make_model(c,cfg,arm)

if __name__=='__main__':unittest.main()
