"""Reconstruct four saved u2012 Adam proposals and evaluate the same inputs.
No gradients computed, no continuation, no reverse chain. Existing window/cap.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import sys,time,json,traceback,signal
from pathlib import Path
import worker as audit
base,rt,g,torch,np=audit.base,audit.rt,audit.g,audit.torch,audit.np
OUT=audit.OUT
def main():
    start=time.time();used=json.loads((OUT/'resource_receipt.json').read_text())['GPU_hours']*3600
    window=json.loads((OUT/'window.json').read_text())
    assert start+250<window['science_stop_epoch'] and used+250<window['GPU_seconds_cap']
    folder=OUT/'same_input_step';folder.mkdir(exist_ok=False)
    rt.setup();records=[];receipt=dict(start_epoch=start,status='running',extra_forwards=0,extra_backwards=0,optimizer_reconstructions=0)
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(TimeoutError('bounded follow-up stop')))
    try:
        for task in json.loads((OUT/'protocol.json').read_text())['tasks']:
            assert time.time()<start+220
            row=task['parent'];origin=OUT/task['id'];u=2012
            z,cfg,case,m,o,e,rng,b,core,exact,canonical=base.setup(row)
            state=torch.load(origin/'boundary_u2011.pt',map_location='cpu',weights_only=False)['state']
            v=torch.load(origin/'vectors_u2012.pt',map_location='cpu',weights_only=False)
            b=rt.restore(state,m,o,e,rng);batch=base.inputs(case,cfg,rng,row,u,4)
            historical=next(json.loads(x) for x in (origin/'training.jsonl').read_text().splitlines() if json.loads(x)['update']==u)
            assert [rt.input_hash([x]) for x in batch]==[x['input_hash'] for x in historical['records']]
            rng_hash=g.digest(g.rng_capture(rng,'cuda'));m.train()
            def evaluate(label):
                result=[];model_hash=g.digest(m.state_dict())
                for i,inp in enumerate(batch):
                    audit.check();tok=('same_input',label,i)
                    ctx=g.FrozenTypedContext(m,case['space'].types,inp['edges'],inp['edge_families'],tok)
                    try:
                        with torch.no_grad():
                            graph,logits,p=ctx.predict(inp['query'],inp['f'],inp['t']/1000,tok,details=True)
                            terms=rt.supported_binary_ce(logits,torch.as_tensor(inp['labels'],device='cuda'),m.prior[graph.query_families])
                        receipt['extra_forwards']+=1;ix=np.argsort(inp['ids']);prob=p.cpu().double().numpy()[ix]
                        draws=np.random.default_rng(np.random.SeedSequence([2026111132,row['seed'],u,i])).random((16,len(prob)))
                        assert base.array_digest(draws)==historical['records'][i]['hard_uniform_hash']
                        bits=(draws<prob).astype(np.uint8)
                        values,_,_=exact.value_gradient(prob);cost=base.old.remaining(core,bits)
                        risks=np.zeros(5);risks[[0,2,3]]=values;risks[[1,4]]=cost.mean(0)
                        np.savez_compressed(folder/f'{task["id"]}_{label}_i{i}.npz',probability=prob,labels=inp['labels'][ix],ids=inp['ids'][ix],families=inp['f'][ix])
                        result.append(dict(index=i,t=inp['t'],input_hash=rt.input_hash([inp]),risks=risks.tolist(),
                                           expected_edges=float(prob.sum()),CE=float(terms.sum()/case['space'].total)))
                    finally:ctx.close()
                assert model_hash==g.digest(m.state_dict()) and rng_hash==g.digest(g.rng_capture(rng,'cuda'))
                return result
            before=evaluate('before');params=list(m.parameters());names=[n for n,p in m.named_parameters()]
            assert names==v['names'];theta=[p.detach().clone() for p in params];offset=0
            for p,size in zip(params,v['sizes']):
                grad=v['gradient'][offset:offset+size].reshape_as(p).to('cuda');offset+=size
                p.grad=grad if p in o.state else None
                if p not in o.state: assert not torch.count_nonzero(grad)
            o.step();receipt['optimizer_reconstructions']+=1
            actual=torch.cat([(p.detach()-old).flatten() for p,old in zip(params,theta)]).cpu()
            difference=float((actual-v['displacement']).abs().max())
            # Same CUDA optimizer and stored gradients should reconstruct exactly;
            # this does not claim neural recomputation is bitwise repeatable.
            assert torch.equal(actual,v['displacement']),f'optimizer reconstruction mismatch {difference}'
            after=evaluate('after')
            records.append(dict(task=task['id'],update=u,before=before,after=after,
                                saved_displacement_exact=True,max_displacement_difference=difference,
                                parameters_only_diagnostic_no_resumable_checkpoint=True))
            rt.save(folder/'results.json',records)
            del m,o,e,b,core,exact,params,theta,actual,v,state,z
            torch.cuda.empty_cache()
        receipt['status']='completed'
    except BaseException as exc:receipt.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
    finally:
        torch.cuda.synchronize();receipt.update(end_epoch=time.time(),seconds=time.time()-start,
                      peak_GPU_MiB=torch.cuda.max_memory_allocated()/2**20)
        rt.save(folder/'receipt.json',receipt);print(json.dumps(receipt),flush=True)
    return 0 if receipt['status']=='completed' else 1
if __name__=='__main__':raise SystemExit(main())
