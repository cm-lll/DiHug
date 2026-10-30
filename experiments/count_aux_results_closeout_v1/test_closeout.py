import importlib.util, pathlib, tempfile, time, unittest
import build as B

class CloseoutTests(unittest.TestCase):
    def test_original_dataset_bug_reproduced(self):
        rows=[{'job':{'dataset':'PubMed','train_seed':2026102912,'arm':'CE'}}]
        with self.assertRaises(KeyError):{(x['dataset'],x['train_seed'],x['arm']):x for x in rows}
        self.assertEqual({(x['job']['dataset'],x['job']['train_seed'],x['job']['arm']):x for x in rows}.keys(),{('PubMed',2026102912,'CE')})
    def test_real_task_identities(self):
        for d in B.SEEDS:
            for s in B.SEEDS[d]:
                for a in ('CE','CE_COUNT'):
                    j=dict(dataset=d,train_seed=s,arm=a,id=f'{d}_s{s}_{a}',directory=f'{d}_s{s}_{a}')
                    self.assertEqual(B.task_identity(dict(job=j)),(d,s,a))
    def test_unknown_identity_rejected(self):
        with self.assertRaises(ValueError):B.task_identity(dict(job=dict(dataset='Unknown',train_seed=2026102912,arm='CE')))
    def test_path_identity_rejected(self):
        with self.assertRaises(ValueError):B.task_identity(dict(job=dict(dataset='PubMed',train_seed=2026102912,arm='CE',id='IMDB_s2026102911_CE')))
    def test_mean_absolute_error_before_mean(self):
        self.assertEqual(B.strict_mean([abs(900-926),abs(600-926)]),176)
        self.assertEqual(B.strict_mean([abs(900-926),abs(952-926)]),26)
        self.assertEqual(abs(B.strict_mean([900,952])-926),0)
        self.assertEqual(B.strict_mean([abs(2862-1937),abs(1014-1937)]),924)
        self.assertEqual(abs(B.strict_mean([2862,1014])-1937),1)
    def test_null_is_not_zero(self):self.assertIsNone(B.strict_mean([1,None]))
    def test_missing_samples_not_averaged(self):
        rows={2026102821:4};self.assertIsNone(B.strict_mean([rows.get(s) for s in B.SAMPLES]))
    def test_small_float_tail_is_logged(self):
        delta=[];self.assertTrue(B.equivalent({'x':.19162201245682164},{'x':.19162201245682162},differences=delta));self.assertEqual(len(delta),1)
    def test_material_mismatch_rejected(self):self.assertFalse(B.equivalent({'x':.19},{'x':.20}))
    def test_hash_not_tolerated(self):self.assertFalse(B.equivalent({'sha':'abc'},{'sha':'abd'}))
    def test_fixed_finalizer_executes_without_models(self):
        # Execute actual patched main against tiny saved records, not a model.
        p=pathlib.Path(__file__).parents[1]/'typed_structure_count_aux_length_v1/finalize.py'
        import sys
        sys.path.insert(0,str(B.SOURCE))
        mod=B.module(p,'patched_delivery')
        with tempfile.TemporaryDirectory(dir=str(B.REPO/'results/count_aux_closeout_cpu_tmp')) as tmp:
            out=pathlib.Path(tmp);run=out/'run';run.mkdir()
            jobs=mod.jobs();mod.jobs=lambda:jobs;mod.source_binding=lambda:({},'test_no_models')
            for j in jobs['training']:
                folder=run/j['directory'];folder.mkdir();B.save(folder/'summary.json',dict(status='complete',completed_updates=j['updates']))
            B.save(run/'activation.json',dict(start_epoch=time.time()))
            B.save(run/'launcher_summary.json',dict(status='complete',primary_failures=[]))
            B.save(run/'unified_evaluation.json',dict(inventory=[],paired_means=[],length_means=[]))
            result=mod.main(run,time.time()+60)
            self.assertEqual(result['status'],'complete');self.assertTrue((out/'REPORT.md').exists());self.assertTrue((out/'review_metadata.zip').exists())

if __name__=='__main__':unittest.main(verbosity=2)
