import unittest,pathlib,tempfile,json,sys,random,subprocess
sys.path.insert(0,str(pathlib.Path(__file__).parent/'package'))
import numpy as np
import torch
from admission import dictionaries,assignment,sha
from experiment_contract import reset_training_streams

class Revision(unittest.TestCase):
    def test_external_dictionary_receipt_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=pathlib.Path(tmp);approval={'dictionaries':{}}
            for d in ['IMDB','PubMed']:
                p=root/('dictionary_'+d);p.mkdir();r=dict(status='passed',K=[1],offsets=[0,1]);(p/'selection.json').write_text(json.dumps(r))
                for name in ['final_basis.npz','panel.json','frozen_splits.json']:(p/name).write_bytes(b'fixture')
                approval['dictionaries'][d]=dict(files={x.name:sha(x) for x in p.iterdir()},K=[1],offsets=[0,1],dimension=1)
            dictionaries(root,approval)
            for name in ['final_basis.npz','selection.json','panel.json','frozen_splits.json']:
                p=root/'dictionary_IMDB'/name;before=p.read_bytes();p.write_bytes(before+b' ')
                with self.assertRaises(ValueError):dictionaries(root,approval)
                p.write_bytes(before)
    def test_random_reset_independent_of_construction(self):
        a,proof=reset_training_streams(9,'IMDB','cpu');sample=(random.random(),np.random.random(),torch.rand(3),a.random(3));torch.nn.Linear(153,71);random.random();np.random.random()
        b,proof2=reset_training_streams(9,'IMDB','cpu');self.assertEqual(proof,proof2);self.assertEqual(sample[0],random.random());self.assertEqual(sample[1],np.random.random());self.assertTrue(torch.equal(sample[2],torch.rand(3)));np.testing.assert_array_equal(sample[3],b.random(3))
    def test_assignment_is_execution_queues(self):
        cost={'cells':{d+'_'+a:dict(normal2_seconds=i+1,sampling_step_seconds=.01,validation_save_seconds=1) for i,(d,a) in enumerate((d,a) for d in ['IMDB','PubMed'] for a in 'OABC')}}
        for n in [1,2,3,7,8]:
            queues,loads=assignment(cost,500,list(range(n)));self.assertEqual(sum(map(len,queues)),8)
            for q,load in zip(queues,loads):self.assertEqual(load,sum(500*cost['cells'][d+'_'+a]['normal2_seconds']+21 for d,a in q))
    def test_worker_direct_entry_rejected(self):
        r=subprocess.run([sys.executable,str(pathlib.Path(__file__).parent/'package/worker.py')],capture_output=True,text=True,timeout=15);self.assertNotEqual(r.returncode,0);self.assertIn('--launch-fd',r.stderr)
    def test_missing_summary_does_not_average_success(self):
        from summarize import collect
        with tempfile.TemporaryDirectory() as tmp:r=collect(tmp)
        self.assertEqual(r['evaluated_graphs'],0);self.assertEqual(r['status'],'incomplete');self.assertEqual(len(r['cells']),8);self.assertTrue(all(x['mean'] is None for x in r['contrasts']))

if __name__=='__main__':unittest.main()
