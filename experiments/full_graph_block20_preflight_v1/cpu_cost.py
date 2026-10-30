"""CPU feature/resource accounting; no fitting, NNLS, model or GPU calls."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,json,pathlib,resource,time
import numpy as np
import block20 as B
from features import extract,digest
from hsgd.bw_preflight.common import plan_for,query_inventory

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);a=ap.parse_args();out={}
    for dataset in ('PubMed','IMDB'):
        c,cfg=B.load_case(dataset);inp=B.pair_inputs(c,cfg,0,np.random.default_rng(cfg['seed']+71),500)[0];rows=[]
        for label,edges,families in [('clean_reference',c['clean'],c['space'].encode(c['clean'])[0]),('ordinary_t500',inp['edges'],inp['edge_families'])]:
            start=time.perf_counter();f=extract(c['space'].types,edges,families,c['space'].pairs);elapsed=time.perf_counter()-start
            rows.append(dict(source=label,nodes=len(c['space'].types),edges=edges.shape[1],Xt_hash=B.digest(edges),feature_sha256=digest(f),shape=list(f.shape),bytes=f.nbytes,seconds=elapsed,finite=bool(np.isfinite(f).all()),nonnegative=bool((f>=0).all())))
        plan=plan_for(c,[cfg['seed']+91,500,61],20)
        out[dataset]=dict(features=rows,sampling_plan_inventory=query_inventory(c,plan),full_graph_roles='not fitted; no dictionary or NNLS benchmark',full_dictionary_candidate_fits=64,full_dictionary_final_fits=4,
            seven_state_FP64_feature_bytes=len(c['space'].types)*630*8*7,expected_supervised_slots_per_normal2=2*plan.query_total/20,
            expected_12000_update_slot_evaluations=1200*plan.query_total,expected_per_slot_supervisions_12000=1200,
            formal_training=0,models_constructed=0,nnls_calls=0,nmf_fits=0)
    out['cpu_peak_rss_mib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    pathlib.Path(a.output).write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(json.dumps(out,indent=2))

if __name__=='__main__':main()
