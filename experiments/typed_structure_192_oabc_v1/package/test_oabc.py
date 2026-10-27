"""Small interface fixtures only; no real dictionary substitution or updates."""
import unittest,copy,tempfile,json,pathlib,time
from unittest.mock import patch
import numpy as np
import torch
from test_engineering import toy,config
from features import extract
from model_variants import StructuralDenoiser,copy_common
from role_dictionary import feature_groups,node_folds,weighted_rows,masked_score,choose,encode_typed,seed32
from normal2 import normal2,pair_inputs
from hsgd.computational import FrozenTypedContext
from hsgd.bw_preflight.common import plan_for
from experiment_contract import read_dictionary,scientific_gate,recommend_tier

torch.set_num_threads(1)
def fixture():
    c=toy();width=9*(1+6);rng=np.random.default_rng(173);h=rng.uniform(.1,1,(3,width));h/=np.linalg.norm(h,axis=1)[:,None]
    return c,dict(basis=h,offsets=np.array([0,1,3]))
def prediction(c,m,edges=None):
    edges=c['clean'] if edges is None else edges;f=c['space'].encode(edges)[0];ctx=FrozenTypedContext(m,c['space'].types,edges,f,('test',0))
    try:
        ids,qf,q=plan_for(c,13,1).block(0);return ids,ctx.predict(q,qf,.5,('test',0),details=True)
    finally:ctx.close()

class DictionaryProtocol(unittest.TestCase):
    def test_feature_ancestry_complete_disjoint(self):
        groups,sides,perm=feature_groups();self.assertEqual(len(groups),30);self.assertEqual([len(x) for x in groups],[21]*30)
        self.assertEqual(sorted(sum(groups,[])),list(range(630)));self.assertFalse(set(sides[0])&set(sides[1]));self.assertEqual([len(x) for x in sides],[315,315])
        for g in groups:self.assertTrue(set(g)<=set(sides[0]) or set(g)<=set(sides[1]))
    def test_folds_stable_all_states_and_caps(self):
        for name,counts,want in [('IMDB',[48,56,78,10],[8,8,8,5]),('PubMed',[40,74,66,12],[8,8,8,6])]:
            types=np.repeat(np.arange(4),counts);folds=node_folds(types,name)
            self.assertEqual([folds[i]['Kmax'] for i in range(4)],want)
            for typ,row in folds.items():
                a,b=row['folds'];self.assertFalse(set(a)&set(b));self.assertEqual(set(a)|set(b),set(np.flatnonzero(types==typ)))
                self.assertTrue(np.array_equal(a,node_folds(types,name)[typ]['folds'][0]))
    def test_weighted_objective_equivalence(self):
        f=[np.ones((3,4))*(i+1) for i in range(7)];m=weighted_rows(f,np.arange(3));loss=float((m*m).sum())/(8*3*4)
        target=.25*np.mean(f[0]**2)+.125*sum(np.mean(x*x) for x in f[1:]);self.assertAlmostEqual(loss,target)
    def test_masked_encoding_never_observes_target(self):
        from role_dictionary import nnls_encode
        h=np.array([[1.,0.,.5,0.],[0.,1.,0.,.5]]);f=[np.array([[.3,.8,.1,.2]]) for _ in range(7)];sides=[np.array([0,1]),np.array([2,3])];seen=[]
        def observed(x,b):seen.append(x.copy());return nnls_encode(x,b)
        with patch('role_dictionary.nnls_encode',observed):a=masked_score(f,np.array([0]),h,sides)
        f2=[x.copy() for x in f]
        for x in f2:x[:,2:]+=7
        with patch('role_dictionary.nnls_encode',observed):b=masked_score(f2,np.array([0]),h,sides)
        for i in range(7):np.testing.assert_array_equal(seen[i],seen[14+i])
        self.assertNotEqual(a[0]['weighted_mse'],b[0]['weighted_mse'])
    def test_rank_tie_and_failure_cannot_be_dropped(self):
        self.assertEqual(choose([dict(K=1,valid=True,mean_mse=.1+5e-13),dict(K=2,valid=True,mean_mse=.1)]),1)
        with self.assertRaises(ValueError):choose([dict(K=1,valid=True,mean_mse=.1),dict(K=2,valid=False,mean_mse=None)])
    def test_dynamic_segments_and_permutation(self):
        h=np.eye(4);offsets=np.array([0,1,4]);types=np.array([0,1,1]);f=np.array([[2.,0.,0.,0.],[0.,3.,4.,1.],[0.,1.,2.,3.]])
        c,_=encode_typed(f,h,offsets,types);np.testing.assert_array_equal(c,f);self.assertEqual(c.sum(1)[1],8)
        p=np.array([2,0,1]);cp,_=encode_typed(f[p],h,offsets,types[p]);np.testing.assert_array_equal(cp,c[p])
    def test_seed_streams_distinct(self):
        a=seed32(2026102831,dataset='IMDB',time=100,purpose='dictionary_forward_noise');b=seed32(2026102831,dataset='PubMed',time=100,purpose='dictionary_forward_noise');self.assertNotEqual(a,b)
    def test_dictionary_failure_and_review_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=pathlib.Path(tmp);d=p/'dictionary_IMDB';d.mkdir();(d/'selection.json').write_text(json.dumps(dict(status='blocked_incomplete_candidate_comparison')))
            with self.assertRaises(RuntimeError):read_dictionary(p,'IMDB')
            (p/'ledger.json').write_text(json.dumps(dict(deadline_epoch=time.time()+60)));(p/'approval.json').write_text('{}')
            with self.assertRaises(PermissionError):scientific_gate(p/'ledger.json',p/'approval.json',p/'absent',p,'hash')
    def test_cost_requires_eight_cells(self):
        with self.assertRaises(ValueError):recommend_tier(dict(complete_eight_cells=False),10000,8)

class Models(unittest.TestCase):
    def test_variable_dictionary_state_load(self):
        c,basis=fixture();left=StructuralDenoiser(c,config(),'C',basis)
        with torch.no_grad():left.role_projection.weight.fill_(.03)
        right=StructuralDenoiser(c,config(),'C',basis);right.load_state_dict(copy.deepcopy(left.state_dict()),strict=True)
        for k,v in left.state_dict().items():self.assertTrue(torch.equal(v,right.state_dict()[k]))
        torch.testing.assert_close(prediction(c,left)[1][1],prediction(c,right)[1][1],atol=0,rtol=0)

    def test_node_permutation_nonzero_projection(self):
        from hsgd.bw_preflight.common import make_case
        c,basis=fixture();m=StructuralDenoiser(c,config(),'C',basis)
        with torch.no_grad():m.refex_projection.weight.fill_(.01);m.role_projection.weight.fill_(.02)
        perm=np.array([2,0,1,5,3,4]);types=np.empty_like(c['space'].types);types[perm]=c['space'].types
        other=make_case(types,np.sort(perm[c['clean']],axis=0),c['space'].pairs)
        ids,(_,_,p)=prediction(c,m);ids2,(_,_,p2)=prediction(other,m)
        _,_,q=plan_for(c,13,1).block(0);mapped=np.sort(perm[q],axis=0);family,local=other['space'].encode(mapped);mapped_ids=other['space'].offsets[family]+local
        lookup={int(i):j for j,i in enumerate(ids2)};order=[lookup[int(i)] for i in mapped_ids]
        torch.testing.assert_close(p,p2[order],atol=2e-6,rtol=1e-5)
    def test_O_and_BC_have_no_hidden_CN(self):
        c,basis=fixture()
        for arm in 'OABC':
            m=StructuralDenoiser(c,config(),arm,basis)
            self.assertEqual(m.common_neighbor is not None,arm=='A')
            self.assertEqual(hasattr(m,'refex_projection'),arm in 'BC');self.assertEqual(hasattr(m,'role_projection'),arm=='C')
            if arm!='A':self.assertEqual(m.cn_feature_width,0)
            if arm=='O':
                with patch('model_variants.extract',side_effect=AssertionError('O used ReFeX')):prediction(c,m)
    def test_actual_copy_zero_projection_and_optimizer_membership(self):
        c,basis=fixture();torch.manual_seed(42);a=StructuralDenoiser(c,config(),'A');models={}
        for arm in 'OBC':
            m=StructuralDenoiser(c,config(),arm,basis);copy_common(a,m);models[arm]=m
            opt=torch.optim.AdamW(m.parameters());self.assertEqual({id(p) for g in opt.param_groups for p in g['params']},{id(p) for p in m.parameters()})
        _,(_,lo,po)=prediction(c,models['O'])
        for arm in 'BC':
            _,(_,l,p)=prediction(c,models[arm]);self.assertTrue(torch.equal(lo,l));self.assertTrue(torch.equal(po,p))
        self.assertEqual(models['C'].role_projection.in_features,3)
    def test_normal2_gradient_without_update_and_basis_frozen(self):
        c,basis=fixture();m=StructuralDenoiser(c,config(),'C',basis);inputs=pair_inputs(c,dict(seed=9,T=1000),1,np.random.default_rng(11),500);opt=torch.optim.AdamW(m.parameters());before={k:v.clone() for k,v in m.state_dict().items()}
        with patch.object(opt,'step',side_effect=AssertionError('no engineering updates authorized by this unit test')):rec=normal2(m,opt,c,inputs,1000,'cpu',False,{})
        self.assertEqual(rec['optimizer_steps'],0);self.assertIsNone(m.role_basis.grad)
        for k,v in m.state_dict().items():self.assertTrue(torch.equal(v,before[k]))
        self.assertIsNotNone(m.role_projection.weight.grad);self.assertTrue(torch.isfinite(m.role_projection.weight.grad).all())
    def test_cache_mutation_refused_and_independent_Xt(self):
        c,basis=fixture();m=StructuralDenoiser(c,config(),'C',basis);prediction(c,m);self.assertEqual(m.feature_calls,1);prediction(c,m);self.assertEqual(m.feature_calls,2)
        with torch.no_grad():m.role_offsets[1]+=1
        with self.assertRaises(ValueError):prediction(c,m)
    def test_edge_order_and_endpoint_canonicalization(self):
        c,basis=fixture();m=StructuralDenoiser(c,config(),'C',basis)
        with torch.no_grad():m.refex_projection.weight.fill_(.01);m.role_projection.weight.fill_(.02)
        _,(_,l,p)=prediction(c,m);reversed_edges=np.sort(c['clean'][::-1,::-1],axis=0);_,(_,l2,p2)=prediction(c,m,reversed_edges)
        torch.testing.assert_close(l,l2,atol=2e-6,rtol=1e-5);torch.testing.assert_close(p,p2,atol=2e-6,rtol=1e-5)

class PublicGates(unittest.TestCase):
    def test_public_launcher_refuses_before_devices_or_output(self):
        import subprocess,sys
        with tempfile.TemporaryDirectory() as tmp:
            root=pathlib.Path(tmp);(root/'ledger.json').write_text(json.dumps(dict(deadline_epoch=time.time()+60)));(root/'approval.json').write_text('{}');(root/'protocol.json').write_text('{}')
            cmd=[sys.executable,str(pathlib.Path(__file__).parent/'launch.py'),'formal','--ledger',str(root/'ledger.json'),'--approval',str(root/'approval.json'),'--protocol',str(root/'protocol.json'),'--dictionary-dir',str(root/'absent'),'--cost',str(root/'missing'),'--output',str(root/'output'),'--gpu-uuids','GPU-not-used']
            r=subprocess.run(cmd,capture_output=True,text=True,timeout=15)
            self.assertNotEqual(r.returncode,0);self.assertIn('review + user continuation missing',r.stderr);self.assertFalse((root/'output').exists())

    def test_evaluation_identity_and_contrasts(self):
        from evaluation import evaluate,paired_table
        c=toy();r=evaluate(c,c['clean'],['type0','type1'])
        self.assertEqual(list(r['primary'].values()),[0.,0.,0.])
        rows={(d,a,s):r for d in ['IMDB','PubMed'] for a in 'OABC' for s in [2026102821,2026102822]}
        differences=paired_table(rows);self.assertEqual(len(differences),20)
        self.assertTrue(all(not any(row['primary'].values()) for row in differences))

    def test_entry_imports_do_not_launch(self):
        import worker,launch
        self.assertTrue(callable(worker.main));self.assertTrue(callable(launch.main))

if __name__=='__main__':unittest.main()
