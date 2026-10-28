"""Only CPU engineering fixtures; no denoiser, fitting, optimizer or sampling."""
import copy,hashlib,itertools,json,pathlib,sys,tempfile,time,unittest
from unittest.mock import patch
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'package'))
import summarize as summary
import launch,admission
from experiment_contract import recommend_tier
from restore_checks import restoration_passes
from evaluation import evaluate
from hsgd.bw_preflight.common import make_case

NAMES=['Director','Actor','Movie','Genre']
def fixture():
    return make_case(np.arange(4),np.array([[0,1,2],[1,2,3]]),tuple(itertools.combinations_with_replacement(range(4),2)))
def loader(dataset):return fixture(),{'datasets':{dataset:{'type_names':NAMES}}}
def value(edges=None):
    c=fixture();m=evaluate(c,c['clean'] if edges is None else edges,NAMES)
    return summary.graph_summary(m,m['reference_metrics']),m
def save_graph(root,dataset,arm,seed):
    c=fixture();folder=root/(dataset+'_'+arm);folder.mkdir(exist_ok=True)
    (folder/'final.pt').write_bytes(b'ENGINEERING FIXTURE ONLY; NOT A MODEL')
    (folder/'summary.json').write_text(json.dumps(dict(status='complete',completed_updates=1000,completed_chains=2)))
    sample=folder/('sample_'+str(seed));sample.mkdir()
    np.savez(sample/'graph.npz',types=c['space'].types,schema=c['space'].pairs,edges=c['clean'],families=c['space'].encode(c['clean'])[0])
    (sample/'binding.json').write_text(json.dumps(dict(seed=seed,graph_sha256=summary.sha(sample/'graph.npz'),checkpoint_sha256=summary.sha(folder/'final.pt'),binding={'arm':arm})))
def record():
    good={'passes':True}
    return dict(optimizer_exact=True,rng_exact=True,loss=good,inputs=[{k:good for k in ['logits','p0','loss']} for _ in range(2)],**{g:{'x':good} for g in ['gradients','parameters','optimizer']})
def dummy_script(root,body):
    package=root/'package';package.mkdir();script=root/'summarize.py';script.write_text(body)
    out=root/'out';out.mkdir();return package,out,{'summarize_sha256':summary.sha(script)}

class SummaryRevision(unittest.TestCase):
    def test_path_labels_in_full_summary_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=pathlib.Path(tmp);save_graph(root,'IMDB','O',summary.SEEDS[0])
            with patch.object(summary,'load_case',loader):r=summary.collect(root)
        raw=r['cells']['IMDB_O']['graphs'][str(summary.SEEDS[0])]['summary']['raw']
        self.assertEqual([raw['MP%d_total'%i] for i in [1,2,3]],[3,2,1]);self.assertNotIn('MP4_total',raw)
    def test_absolute_error_before_mean(self):
        a,m=value();b=copy.deepcopy(a);a['raw']['edges']=2;b['raw']['edges']=4
        for x in [a,b]:x['absolute_error']['edges']=abs(x['raw']['edges']-x['reference']['edges'])
        mean=summary.strict_mean([a,b]);self.assertEqual(mean['raw']['edges'],3);self.assertEqual(mean['absolute_error']['edges'],1)
    def test_pair_raw_and_error_are_separate(self):
        a,_=value();b=copy.deepcopy(a);a['raw']['edges']=4;a['absolute_error']['edges']=1;b['raw']['edges']=2;b['absolute_error']['edges']=1
        r=summary.paired(a,b,a);self.assertEqual(r['raw_difference']['edges'],2);self.assertEqual(r['absolute_error_difference']['edges'],0)
    def test_zero_closure_denominator_preserved(self):
        a,m=value();self.assertIsNone(a['raw']['closure_path_ratio']);self.assertEqual(a['raw']['closure_path_denominator'],0)
        self.assertEqual(a['undefined']['raw']['closure_path_ratio'],'undefined_zero_closure_denominator')
    def test_null_and_missing_metric_propagation(self):
        a,_=value();b=copy.deepcopy(a);b['raw']['closure_path_ratio']=.5;b['absolute_error']['closure_path_ratio']=.5
        del b['raw']['edges'];r=summary.strict_mean([a,b]);p=summary.paired(a,b,a)
        self.assertIsNone(r['raw']['closure_path_ratio']);self.assertIsNone(r['raw']['edges']);self.assertIsNone(p['raw_difference']['edges'])
        self.assertIsNone(p['raw_difference']['closure_path_ratio'])
    def test_missing_group_not_dropped(self):
        r=summary.strict_mean([{'raw':{'a':1}},{'raw':{'a':2},'extra':{'b':4}}]);self.assertIsNone(r['extra']['b'])
    def test_missing_graph_not_average_successes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=pathlib.Path(tmp);save_graph(root,'IMDB','O',summary.SEEDS[0]);(root/'IMDB_O/summary.json').write_text('{"status":"failed","error":"fixture failure"}')
            with patch.object(summary,'load_case',loader):r=summary.collect(root)
        self.assertEqual(r['evaluated_graphs'],1);self.assertEqual(r['status'],'incomplete');self.assertEqual(len(r['cells']),8)
        self.assertIsNone(r['cells']['IMDB_O']['mean']);self.assertEqual(r['cells']['IMDB_O']['worker_status']['error'],'fixture failure')
        self.assertTrue(all(c['mean'] is None for c in r['contrasts']));self.assertEqual(r['cells']['PubMed_C']['worker_status']['status'],'not_started')
    def test_complete_fixture_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=pathlib.Path(tmp)
            for d,a,s in itertools.product(['IMDB','PubMed'],'OABC',summary.SEEDS):save_graph(root,d,a,s)
            with patch.object(summary,'load_case',loader):r=summary.collect(root)
        self.assertEqual(r['status'],'complete');self.assertEqual(r['evaluated_graphs'],16)
        self.assertTrue(all(c['mean']['raw_difference']['MP3_total']==0 for c in r['contrasts']))
        json.dumps(r,allow_nan=False)
    def test_mp_composition_pair_and_mean(self):
        a,m=value(np.array([[0,0,0],[1,2,3]]));b,_=value()
        for i in [2,3]:self.assertEqual(a['distribution_error']['MP%d_composition_TV'%i],m['comparison']['mp_frequency_TV'][i-1])
        self.assertGreater(a['distribution_error']['MP2_composition_TV'],0);self.assertGreater(a['distribution_error']['MP3_composition_TV'],0)
        p=summary.paired(a,b,b);mean=summary.strict_mean([p,p]);self.assertEqual(mean['distribution_error_difference']['MP2_composition_TV'],p['distribution_error_difference']['MP2_composition_TV'])
    def test_w1_independent_compatibility(self):
        a,m=value(np.array([[0,1],[1,2]]));self.assertGreater(a['distribution_error']['degree_W1_incident4'],0)
        self.assertAlmostEqual(a['distribution_error']['degree_W1_all10_compat'],.4*a['distribution_error']['degree_W1_incident4'],places=14)
    def test_reference_zero_triangle_is_count_error_not_schema_ban(self):
        a,m=value(np.array([[0,0,1],[1,2,2]]));key='typed_triangle_(0, 1, 2)'
        self.assertEqual(a['raw'][key],1);self.assertEqual(a['reference'][key],0);self.assertEqual(a['absolute_error'][key],1)
        self.assertTrue(m['schema_legal']);self.assertIn('type_Genre_relation_9_zero',a['absolute_error'])
    def test_exact_optimizer_negative_inside_tolerance(self):
        import torch
        from probe import equal,metric,flatten
        r=record();self.assertTrue(restoration_passes(r))
        saved={'state':{0:{'step':torch.tensor(2.),'exp_avg':torch.ones(3),'exp_avg_sq':torch.ones(3)}},'param_groups':[{'params':[0],'lr':.0002}]}
        restored=copy.deepcopy(saved);restored['state'][0]['exp_avg'][1]=torch.nextafter(torch.tensor(1.),torch.tensor(2.))
        a,b=flatten(saved),flatten(restored)
        r['optimizer']={k:metric(a[k],b[k]) for k in a};r['optimizer_exact']=equal(saved,restored)
        self.assertTrue(all(x['passes'] for x in r['optimizer'].values()))
        self.assertEqual(sum(int((a[k]!=b[k]).sum()) for k in a),1);self.assertFalse(r['optimizer_exact']);self.assertFalse(restoration_passes(r))
    def test_original_restore_gates_preserved(self):
        for field in ['rng_exact','loss','inputs','gradients','parameters','optimizer']:
            r=record()
            if field=='rng_exact':r[field]=False
            elif field=='loss':r[field]={'passes':False}
            elif field=='inputs':r[field][0]={'logits':{'passes':False},'p0':{'passes':True},'loss':{'passes':True}}
            else:r[field]={'x':{'passes':False}}
            self.assertFalse(restoration_passes(r),field)
    def test_zero_single_parameter_update_is_observation(self):
        import torch
        from normal2 import update_observation
        before=torch.ones(2);after=before.clone()
        r=update_observation(before,after,True);self.assertEqual(r['changed'],0);self.assertFalse(r['observed_parameter_changed'])
        after=before+torch.tensor(1e-10,dtype=torch.float32)
        self.assertTrue(torch.equal(before,after));self.assertEqual(update_observation(before,after,True)['update_note'],'watched_parameter_unchanged_not_failure')
        self.assertIsNone(update_observation(before,after,False)['observed_parameter_changed'])
        self.assertTrue(update_observation(before,before+1,True)['observed_parameter_changed'])
    def test_nonfinite_observation_still_rejected(self):
        import torch
        from normal2 import update_observation
        for bad in [float('nan'),float('inf')]:
            with self.assertRaises(FloatingPointError):update_observation(torch.ones(1),torch.tensor([bad]),True)
    def test_source_hash_rejected_before_invocation(self):
        with tempfile.TemporaryDirectory() as tmp:
            p,out,approval=dummy_script(pathlib.Path(tmp),'raise AssertionError("must not run")')
            self.assertEqual(admission.verify_summary(p,approval),p.parent/'summarize.py')
            (p.parent/'summarize.py').write_text('print("tampered")')
            with self.assertRaises(ValueError):admission.verify_summary(p,approval)
            r=launch.run_summary(p,out,approval,time.time()+20);self.assertEqual(r['status'],'failed');self.assertFalse((out/'summary_process.log').exists())
    def test_source_hash_rejected_by_launch_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=pathlib.Path(tmp);p,out,a=dummy_script(root,'# fixture')
            ledger=root/'ledger.json';protocol=root/'protocol.json';ap=root/'test_only_receipt.json'
            ledger.write_text(json.dumps({'deadline_epoch':time.time()+60}));protocol.write_text('{}')
            a.update(independent_review='passed',user_continue_confirmed=True,main_W1_definition_confirmed='type_weighted_incident4',ledger_sha256=launch.sha(ledger),protocol_sha256=launch.sha(protocol),source_sha256=hashlib.sha256(b'{}').hexdigest())
            ap.write_text(json.dumps(a));(root/'summarize.py').write_text('# changed')
            with self.assertRaisesRegex(ValueError,'summarize'):launch.gate(p,ledger,ap,protocol,root/'absent',root/'absent')
    def test_real_subprocess_incomplete_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            p,out,a=dummy_script(pathlib.Path(tmp),'import sys,pathlib,json\np=pathlib.Path(sys.argv[sys.argv.index("--output")+1]);p.write_text(json.dumps({"status":"incomplete"}));sys.exit(2)\n')
            r=launch.run_summary(p,out,a,time.time()+10);self.assertEqual(r['status'],'incomplete_summary_saved');self.assertEqual(r['exit_code'],2)
    def test_summary_failure_preserves_primary(self):
        with tempfile.TemporaryDirectory() as tmp:
            p,out,a=dummy_script(pathlib.Path(tmp),'raise RuntimeError("fixture summary failure")')
            evaluation=launch.run_summary(p,out,a,time.time()+10)
            r=launch.final_receipt(out,[{'cell':'IMDB_A','exit_code':7}],{'GPU':['unstarted']},True,[{'stage':'worker','error':'ORIGINAL'}],{'updates':1000},{'start_epoch':time.time()},evaluation)
            self.assertEqual(r['status'],'incomplete');self.assertEqual(r['primary_failures'][0]['error'],'ORIGINAL');self.assertEqual(r['evaluation']['status'],'failed_no_summary_output')
    def test_summary_timeout_and_insufficient_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            p,out,a=dummy_script(pathlib.Path(tmp),'import time;time.sleep(5)')
            r=launch.run_summary(p,out,a,time.time()+2.2);self.assertEqual(r['status'],'timeout')
            r=launch.run_summary(p,out,a,time.time());self.assertEqual(r['status'],'skipped_insufficient_delivery_time')
    def test_lpt_budget_and_launcher_same_function(self):
        cost=dict(complete_eight_cells=True,concurrency_margin=1.25,cells={d+'_'+a:dict(normal2_seconds=i+1,sampling_step_seconds=.1,validation_save_seconds=3) for i,(d,a) in enumerate(itertools.product(['IMDB','PubMed'],'OABC'))})
        self.assertIs(launch.assignment,admission.assignment)
        for n in [7,8]:
            devices=list(range(n));queues,loads=admission.assignment(cost,1000,devices);_,tiers=recommend_tier(cost,1e9,n)
            self.assertEqual(tiers[1]['seconds'],max(loads)*1.25+2100);self.assertEqual(sum(map(len,queues)),8)
            for q,load in zip(queues,loads):self.assertEqual(load,sum(1000*cost['cells'][d+'_'+a]['normal2_seconds']+203 for d,a in q))
            if n==7:self.assertEqual([q for q in queues if len(q)==2],[[('IMDB','A'),('IMDB','O')]])
            else:self.assertTrue(all(len(q)==1 for q in queues))

if __name__=='__main__':unittest.main()
