"""Pure CPU resource-policy fixtures. No model or GPU context."""
import pathlib,sys,copy,unittest
sys.path.insert(0,str(pathlib.Path(__file__).parent/'package'))
from launch import resource_check
class ResourceGate(unittest.TestCase):
    def setUp(self):
        self.ids=['GPU-fixture-'+str(i) for i in range(8)]
        self.rows={u:[str(i),u,'NVIDIA GeForce RTX 2080 Ti','11011','0','1'] for i,u in enumerate(self.ids)}
        self.auth=dict(basis='explicit_user_eight_card_authorization',allowed_uuids=self.ids,allocation_check_complete=True,reservation_conflicts=[],minimum_free_mib=5556)
    def test_one_percent_allowed_and_recorded(self):
        r=resource_check(self.ids,self.auth,self.rows,'','');self.assertEqual(len(r),8);self.assertTrue(all(x['utilization_percent']==1 for x in r))
    def test_empty_process_list_not_permission(self):
        a=copy.deepcopy(self.auth);a['allowed_uuids']=[]
        with self.assertRaises(PermissionError):resource_check(self.ids,a,self.rows,'','')
    def test_reservation_rejected(self):
        a=copy.deepcopy(self.auth);a['reservation_conflicts']=['external reservation']
        with self.assertRaises(PermissionError):resource_check(self.ids,a,self.rows,'','')
    def test_other_process_rejected(self):
        with self.assertRaises(RuntimeError):resource_check(self.ids,self.auth,self.rows,self.ids[0]+', 123, 100','')
        with self.assertRaises(RuntimeError):resource_check(self.ids,self.auth,self.rows,'','123')
    def test_insufficient_memory_rejected(self):
        self.rows[self.ids[0]][3]='5555'
        with self.assertRaises(RuntimeError):resource_check(self.ids,self.auth,self.rows,'','')
    def test_seven_cards_rejected(self):
        with self.assertRaises(PermissionError):resource_check(self.ids[:7],self.auth,self.rows,'','')
    def test_wrong_model_or_missing_card_rejected(self):
        self.rows[self.ids[0]][2]='different GPU'
        with self.assertRaises(RuntimeError):resource_check(self.ids,self.auth,self.rows,'','')
        del self.rows[self.ids[0]]
        with self.assertRaises(RuntimeError):resource_check(self.ids,self.auth,self.rows,'','')
if __name__=='__main__':unittest.main()
