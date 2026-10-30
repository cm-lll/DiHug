"""Final loss engineering check: NO optimizer.step, NO graph sampling."""
import argparse,pathlib,time,traceback
from model_io import fresh,configure
from runtime_contract import save,sha,source_binding
import torch,numpy as np
from probe import rng_get,equal,input_hash
from normal2 import pair_inputs,normal2
from count_aux import count_loss,case_contract,LAMBDA
from hsgd.computational import FrozenTypedContext
from hsgd.losses import supported_binary_ce

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--output',required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cpu');a=p.parse_args();device=a.device
    out=pathlib.Path(a.output);out.mkdir(parents=True,exist_ok=False);r=dict(status='running',optimizer_step_calls=0,formal_updates=0,generated_graphs=0,source_sha256=source_binding()[1]);start=time.time()
    try:
        r['environment']=configure() if device=='cuda' else {'device':'cpu','torch':torch.__version__};torch.set_num_threads(1);seed={'IMDB':2026102911,'PubMed':2026102912}[a.dataset]
        m,opt,case,cfg,rng,proof=fresh(a.dataset,'CE_COUNT',seed,device=device)
        torch.save(dict(model=m.state_dict(),rng=rng_get(rng,device),proof=proof,updates=0),out/'initial.pt')
        r['initial_sha256']=sha(out/'initial.pt');fixed=pair_inputs(case,cfg,1,np.random.default_rng(2026103001),500);r['fixed_input_sha256']=input_hash(fixed)
        inp=fixed[0];token='scale_final';c=FrozenTypedContext(m,case['space'].types,inp['edges'],inp['edge_families'],token)
        try:
            g,l,p0=c.predict(inp['query'],inp['f'],.5,token,details=True)
            ce=supported_binary_ce(l,torch.as_tensor(inp['labels'],device=device),m.prior[g.query_families]).sum()/18336
            aux,cr=count_loss(p0,g.query_families,*case_contract(case,inp));gc=torch.autograd.grad(ce,l,retain_graph=True)[0];ga=torch.autograd.grad(LAMBDA*aux,l)[0]
            r['scale']=dict(ce=float(ce),auxiliary=float(aux),weighted_auxiliary=float(LAMBDA*aux),ce_logits_gradient_l2=float(gc.norm()),weighted_aux_logits_gradient_l2=float(ga.norm()),gradient_ratio=float(ga.norm()/gc.norm()),relation_records={k:v.detach().cpu().tolist() if torch.is_tensor(v) else v for k,v in cr.items()})
        finally:c.close()
        del g,l,p0,ce,aux,gc,ga,c,cr
        # Real full normal2 backward with final formula, without any update.
        before=rng_get(rng,device);r['normal2_backward']=normal2(m,opt,case,fixed,1000,device,step=False)
        r['rng_unchanged']=equal(before,rng_get(rng,device))
        r['aux_path_finite_nonzero']=r['scale']['weighted_aux_logits_gradient_l2']>0 and r['normal2_backward']['gradient_norm']>0
        r['parameters']=proof['parameters'];r['paired_tensor_sha256']=proof['tensor_sha256']
        if not r['rng_unchanged'] or not r['aux_path_finite_nonzero']:raise AssertionError('final gradient/RNG')
        r['status']='passed'
    except BaseException as e:r.update(status='failed',error=repr(e),traceback=traceback.format_exc())
    finally:
        r.update(seconds=time.time()-start,final_formula_updated_continuation_tested=False)
        if torch.cuda.is_initialized():r.update(peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
        save(out/'summary.json',r)
    return 0 if r['status']=='passed' else 2
if __name__=='__main__':raise SystemExit(main())
