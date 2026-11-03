"""Run the targeted fault suite and save a small machine-readable receipt."""
import argparse
import datetime
import json
import pathlib
import resource
import time
import unittest

import test_abort_boundaries as cases
import test_role_validation_boundaries as followup


class Result(unittest.TextTestResult):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.records=[];self.times={}
    def startTest(self,test):
        self.times[test.id()]=time.perf_counter();super().startTest(test)
    def addSuccess(self,test):
        self.records.append(dict(test=test.id(),status='passed',seconds=time.perf_counter()-self.times[test.id()]))
        super().addSuccess(test)
    def addFailure(self,test,err):
        self.records.append(dict(test=test.id(),status='failed',error=str(err[1])))
        super().addFailure(test,err)
    def addError(self,test,err):
        self.records.append(dict(test=test.id(),status='error',error=str(err[1])))
        super().addError(test,err)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);a=parser.parse_args()
    began=time.perf_counter()
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromModule(module)
                             for module in (cases,followup))
    result=unittest.TextTestRunner(verbosity=2,resultclass=Result).run(suite)
    import torch
    receipt=dict(status='passed' if result.wasSuccessful() else 'failed',tests_run=result.testsRun,
      failures=len(result.failures),errors=len(result.errors),records=result.records,
      seconds=time.perf_counter()-began,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
      GPU_used=False,CUDA_initialized=torch.cuda.is_initialized(),CPU_threads=torch.get_num_threads(),
      reference_commit='2f2cb300c1c7efc3dd6a28d462d5cb7584e88919',
      followup_parent=followup.PARENT,followup_evidence=followup.EVIDENCE,
      scope='Retains all ten original fault tests. Eight follow-up tests reproduce both parent bugs and exercise actual O/A/D/E eight-layer layouts on a synthetic 12-node graph, real AdamW/EMA/CE and complete next-update recovery; no formal datasets, trained weights, fitted dictionaries or GPU loaded.',
      at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat())
    p=pathlib.Path(a.output);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    if receipt['CUDA_initialized']:raise RuntimeError('CPU-only test unexpectedly initialized CUDA')
    return 0 if result.wasSuccessful() else 1


if __name__=='__main__':raise SystemExit(main())
