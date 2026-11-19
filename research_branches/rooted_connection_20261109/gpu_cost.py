import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import json,time,traceback,resource
from pathlib import Path
import numpy as np
import torch
from score import RootedScore
from gpu_score import GPUIncremental
OUT=Path(__file__).resolve().parents[2]/'results/rooted_connection_20261109T015142'
def main():
    began=time.time();torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    rows=[]
    try:
        for ds in ['IMDB','PubMed']:
            with np.load(OUT/f'{ds}_cpu_cost_panel.npz') as z:data={k:z[k] for k in z.files}
            core=RootedScore(data['types'],data['query'],data['families'],data['reference'],int(data['families'].max())+1)
            before=torch.cuda.get_rng_state().clone();start=time.time()
            inc=GPUIncremental(core,data['bits']);d=inc.differences();torch.cuda.synchronize()
            elapsed=time.time()-start;error=float(np.max(abs(d-data['conditional_difference'])))
            assert error<1e-10,error
            assert torch.equal(before,torch.cuda.get_rng_state())
            rows.append(dict(dataset=ds,seconds=elapsed,max_abs_CPU_difference=error,peak_CUDA_MiB=torch.cuda.max_memory_allocated()/2**20))
            print(rows[-1],flush=True)
    except BaseException as e:
        (OUT/'gpu_cost_failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc())))
        raise
    finally:
        (OUT/'gpu_cost_receipt.json').write_text(json.dumps(dict(rows=rows,GPU_seconds=time.time()-began,device=torch.cuda.get_device_name(),peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024),indent=2))
if __name__=='__main__':main()
