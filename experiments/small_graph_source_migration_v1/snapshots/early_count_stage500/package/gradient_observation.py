"""Sparse observations of the two normal2 losses; never assigns .grad."""
import torch

class GradientObservation:
    def __init__(self, model):
        self.named = [(n,p) for n,p in model.named_parameters() if p.requires_grad]
        self.ce = {}; self.aux = {}
    def add(self, ce_half, weighted_aux_half):
        params = [p for _,p in self.named]
        for loss, dest in [(ce_half,self.ce),(weighted_aux_half,self.aux)]:
            values = torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
            for (name,p),g in zip(self.named,values):
                if g is not None:
                    value=g.detach().to(device='cpu',dtype=torch.float64)
                    if not torch.isfinite(value).all():raise FloatingPointError('observation gradient nonfinite')
                    if name in dest:dest[name].add_(value)
                    else:dest[name]=value.clone()
    def result(self):
        groups = {}
        for name,p in self.named:
            key=name.split('.')[0]
            groups.setdefault(key,[]).append(name)
        def stats(names):
            c2=a2=dot=0.; elements=0; active_ce=active_aux=0
            for n in names:
                c=self.ce.get(n);a=self.aux.get(n)
                if c is not None:c2+=float(c.square().sum());active_ce+=1
                if a is not None:a2+=float(a.square().sum());active_aux+=1
                if c is not None and a is not None:dot+=float((c*a).sum())
            return dict(ce_l2=c2**.5,weighted_aux_l2=a2**.5,dot=dot,
                        cosine=dot/(c2*a2)**.5 if c2>0 and a2>0 else None,
                        auxiliary_to_ce_l2=(a2/c2)**.5 if c2>0 else None,
                        combined_l2=max(0.,c2+a2+2*dot)**.5,
                        parameter_tensors=len(names),ce_connected=active_ce,aux_connected=active_aux)
        return dict(scope='actual two inputs, sum of half-input gradients; weighted auxiliary lambda=.001, including CE-arm shadow observation',
                    all_parameters=stats([n for n,p in self.named]),
                    disjoint_groups={k:stats(v) for k,v in groups.items()},
                    named_parameter_count=len(self.named),
                    unique_parameter_ids=len({id(p) for n,p in self.named}))
