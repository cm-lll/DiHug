"""Synthetic CPU-only targeted checks; no formal assets or GPU required."""
import datetime
import json
import pathlib
import platform
import resource
import sys
import time
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'experiments/structure_input_v1'))
import test_structure_input as integration
import test_features_cpu as features
from run_cpu_fault_tests import Result
import torch

if __name__ == '__main__':
    begin = time.perf_counter()
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromModule(m)
                              for m in (features, integration))
    result = unittest.TextTestRunner(verbosity=2, resultclass=Result).run(suite)
    receipt = dict(status='passed' if result.wasSuccessful() else 'failed',
                   tests=result.testsRun, records=result.records,
                   seconds=time.perf_counter()-begin,
                   max_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                   torch=torch.__version__, python=platform.python_version(),
                   CPU_threads=torch.get_num_threads(), GPU_used=False,
                   CUDA_initialized=torch.cuda.is_initialized(),
                   synthetic_nodes=12, production_parameter_widths=True,
                   base_commit='175373c83202b6c8b80aa570e0092c4652031ce6',
                   timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   limits=['No real data or fitted dictionary loaded',
                           'No GPU, formal training, full-chain generation or quality evidence',
                           'CPU feature backend does not reuse CN wedges'])
    out = ROOT/'review/structure_input_v1/CPU_TESTS.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n')
    if torch.cuda.is_initialized():
        raise RuntimeError('unexpected CUDA initialization')
    raise SystemExit(0 if result.wasSuccessful() else 1)
