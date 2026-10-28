import unittest,json,pathlib
from build_cost import estimate

class Cost(unittest.TestCase):
    def fixture(self):
        p=json.loads((pathlib.Path(__file__).parent/'proposal.json').read_text());e={};old=[]
        for d in ['IMDB','PubMed']:
            e[d]={'cells':{}}
            for a in 'DE':
                e[d]['cells'][d+'_'+a]=dict(passed=True,completed_steps=2,normal2_seconds=3.,sampling_component_seconds=.6,
                    full_query_components=[dict(refex_seconds=.01,NNLS_seconds=.01)],peak_allocated_mib=3000,peak_reserved_mib=3500,peak_rss_mib=4000)
            old.append(dict(cell=d+'_A',normal2_postwarmup_mean_seconds='3',sample_1_seconds='600',sample_2_seconds='600'))
        return p,e,old
    def test_queue_and_wall_not_sum(self):
        p,e,o=self.fixture();r=estimate(p,e,o)
        self.assertEqual([(x['dataset'],x['seed'],x['arm'],x['slot']) for x in r['queue']],[(x['dataset'],x['seed'],x['arm'],x['slot']) for x in p['queue']])
        self.assertEqual(r['estimated_sum_task_seconds'],8*r['estimated_eight_card_critical_path_seconds'])
        self.assertFalse(r['formal_authorized']);self.assertFalse(r['measured_eight_card_DE_throughput'])
    def test_incomplete_or_missing_cell_rejected(self):
        p,e,o=self.fixture();e['IMDB']['cells']['IMDB_D']['passed']=False
        with self.assertRaises(ValueError):estimate(p,e,o)

if __name__=='__main__':unittest.main()
