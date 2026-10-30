"""One full t500 query + original posterior per supplied engineering cell.
Zero optimizer updates, uniforms, samples, or state commits.
"""
import argparse,time,sys,pathlib,signal
import numpy as np
import torch
from runtime_contract import ROOT,save,read,sha,source_binding
sys.path.insert(0,str(ROOT/'package'))
from model_io import configure,fresh
from normal2 import pair_inputs,sync
from hsgd.computational import FrozenTypedContext
from hsgd.bw_preflight.common import check_prob,schedule
from hsgd.kernel import MarginalKernel
def main():
    p=argparse.ArgumentParser();p.add_argument('--engineering-root',required=True);p.add_argument('--dataset',required=True);p.add_argument('--deadline',type=float,required=True);a=p.parse_args()
    def stop(*args):raise TimeoutError('preparation deadline')
    signal.signal(signal.SIGALRM,stop);signal.alarm(max(1,int(a.deadline-time.time())))
    configure()
    for arm in ('A','C','D','E'):
        folder=pathlib.Path(a.engineering_root)/(a.dataset+'_'+arm);file=folder/'after_engineering.pt';out=folder/'query_cost.json'
        if out.exists():raise FileExistsError(out)
        model,opt,case,cfg,rng,_=fresh(a.dataset,arm,read(ROOT/'protocol.json')['train_seeds'][a.dataset]);del opt
        payload=torch.load(file,map_location='cpu');model.load_state_dict(payload['model']);model.eval()
        inp=pair_inputs(case,cfg,1,np.random.default_rng(2026102901),500)[0]
        expected=np.flatnonzero(np.repeat(case['active'],case['space'].counts));assert np.array_equal(np.sort(inp['ids']),expected)
        token=('engineering-cost',a.dataset,arm);sync('cuda');begin=time.perf_counter()
        ctx=FrozenTypedContext(model,case['space'].types,inp['edges'],inp['edge_families'],token)
        try:
            with torch.no_grad():p0=check_prob(ctx.predict(inp['query'],inp['f'],.5,token),inp['f'],case)
            sync('cuda');forward=time.perf_counter()-begin
            kernel=MarginalKernel(case['prior'],schedule(1000));fam,local=case['space'].encode(inp['edges']);ids=case['space'].offsets[fam]+local
            xt=np.isin(inp['ids'],ids).astype(np.uint8);begin=time.perf_counter();q=kernel.reverse(p0,xt,inp['f'],499,500);posterior=time.perf_counter()-begin
            assert np.isfinite(q).all() and ((q>=0)&(q<=1)).all()
            save(out,dict(dataset=a.dataset,arm=arm,t=500,query_slots=len(inp['ids']),blocks=1,forward_including_context_features_transfers_checks_seconds=forward,
                posterior_seconds=posterior,feature_seconds=model.feature_seconds,encoding_seconds=model.encoding_seconds,transfer_seconds=model.transfer_seconds,
                engineering_checkpoint_sha256=sha(file),source_sha256=source_binding()[1],state_commits=0,new_graphs=0,optimizer_steps=0,
                note='one ordinary forward-noise state; no sampling or full chain; not all-state throughput'))
        finally:ctx.close()
        del model,payload;torch.cuda.empty_cache()
    signal.alarm(0)
if __name__=='__main__':main()
