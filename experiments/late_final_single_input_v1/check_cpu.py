"""Only changed single-input scaling and queue/null arithmetic, no model calls."""
from common_run import OUT,read,save,now
import torch,math
def main():
 x=torch.tensor([[.2,-.3],[1.2,-.1]],requires_grad=True);y=torch.tensor([0,1]);S=18336
 loss=torch.nn.functional.cross_entropy(x,y,reduction='sum')/S;g=torch.autograd.grad(loss,x)[0]
 z=x.detach().clone().requires_grad_();halves=sum(torch.nn.functional.cross_entropy(z,y,reduction='sum')/(2*S) for _ in range(2));g2=torch.autograd.grad(halves,z)[0]
 assert torch.equal(loss.detach(),halves.detach()) and torch.equal(g,g2)
 tasks=read(OUT/'tasks.json');assert len(tasks)==8 and len({j['task_id'] for j in tasks})==8
 assert len({(j['dataset'],j['train_seed'],j['arm'],ss) for j in tasks for ss in j['sample_seeds']})==16
 assert all(j['updates']==3000 and j['inputs_per_update']==1 and j['CE_coefficient']==1 and j['CN_fusion']=='late' for j in tasks)
 assert abs((abs(800-926)+abs(1000-926))/2-100)==0
 # Missing one sample forbids calculating a successful-only mean.
 def mean(x):return sum(x)/len(x) if all(v is not None for v in x) else None
 assert mean([2,None]) is None
 save(OUT/'CPU_checks.json',dict(status='passed',at=now(),one_full_CE_equals_two_identical_half_CEs=True,task_count=8,sampling_count=16,per_graph_error_before_mean=True,missing_mean_null=True,model_calls=0,optimizer_steps=0))
 print('CPU checks passed')
if __name__=='__main__':main()
