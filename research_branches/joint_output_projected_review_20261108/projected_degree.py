"""Exact PB CDF over identity and symmetric pair-rotation coordinates."""
import hashlib,json,itertools
import numpy as np
import torch
from pb_degree import RelationDegreeCDF,_GroupedCDF
class ProjectedDegreeCDF(RelationDegreeCDF):
    def __init__(self,types,schema,prior,clean_edges,*,data_sha256):
        super().__init__(types,schema,prior,clean_edges,data_sha256=data_sha256)
        clean=set((min(int(u),int(v)),max(int(u),int(v))) for u,v in np.asarray(clean_edges).T)
        qu,qv=self.keys//self.n,self.keys%self.n
        labels=np.array([(int(u),int(v)) in clean for u,v in zip(qu,qv)],int)
        rows=[];signs=[];bounds=[];caps=[];targets=[];weights=[];descriptors=[];cursor=0;G=len(self.groups)
        for typ in sorted(set(self.types.tolist())):
            gids=[i for i,g in enumerate(self.groups) if g['node_type']==typ];m=len(gids)
            if not m:continue
            C=m*(m-1)//2;original_multiplicity=1+(m-1)*(m-2)//2
            specifications=[([gid],[1],original_multiplicity) for gid in gids]
            for a,b in itertools.combinations(gids,2):specifications.extend([([a,b],[1,1],1),([a,b],[1,-1],1)])
            for groupids,sgn,multiplicity in specifications:
                cap=sum(self.capacities[g] for g in groupids);node_indices=[];node_signs=[]
                for local in range(self.groups[groupids[0]]['nodes']):
                    ii=np.concatenate([self.indices[self.bounds[g][0]+local,:self.capacities[g]] for g in groupids])
                    ss=np.concatenate([np.full(self.capacities[g],s,int) for g,s in zip(groupids,sgn)])
                    node_indices.append(ii);node_signs.append(ss)
                indices=np.array(node_indices);ss=np.array(node_signs);target=np.where(ss==1,labels[indices],1-labels[indices]).sum(1)
                nodes=len(indices);rows.extend(indices);signs.extend(ss);bounds.append((cursor,cursor+nodes));cursor+=nodes
                caps.append(cap);targets.append(np.cumsum(np.bincount(target,minlength=cap+1)/nodes)[:cap])
                weights.append(multiplicity/(G*(1+C)))
                descriptors.append(dict(node_type=typ,relations=[self.groups[g]['relation'] for g in groupids],signs=sgn,capacity=cap,nodes=nodes,weight=weights[-1],negative_offset=sum(self.capacities[g] for g,s in zip(groupids,sgn) if s<0)))
        width=max(caps,default=0);self.proj_indices=np.zeros((cursor,width),int);self.proj_valid=np.zeros((cursor,width),bool);self.proj_negative=np.zeros((cursor,width),bool)
        for i,(ii,ss) in enumerate(zip(rows,signs)):
            self.proj_indices[i,:len(ii)]=ii;self.proj_valid[i,:len(ii)]=True;self.proj_negative[i,:len(ii)]=ss<0
        self.proj_target=np.zeros((len(caps),width))
        for i,v in enumerate(targets):self.proj_target[i,:len(v)]=v
        self.proj_bounds=tuple(bounds);self.proj_caps=tuple(caps);self.proj_weights=np.array(weights);self.projections=descriptors;self._projected_cache={}
        assert not weights or abs(sum(weights)-1)<1e-12
        self.binding.update(version='identity_pair_rotation_CDF_v1',coefficient=1.,replaces_PB=True,extra_PB=False,
            orientations='identity plus all single-pair45degree rotations, uniform; no fitted projection',
            type_weight='original active relation/type group count m_type / total G',projections=descriptors)
        self.binding_sha256=hashlib.sha256(json.dumps(self.binding,sort_keys=True).encode()).hexdigest()
    def __call__(self,p,query,families):
        super().__call__(p,query,families)  # original semantic/coverage/support checks, not added to loss
        if not self.proj_caps:return p.sum()*0,p.new_empty(0)
        q=np.asarray(query);order=np.argsort(np.minimum(*q)*self.n+np.maximum(*q));key=(str(p.device),p.dtype)
        if key not in self._projected_cache:
            self._projected_cache[key]=tuple(torch.as_tensor(v,device=p.device,dtype=p.dtype if i>=3 else None) for i,v in enumerate([self.proj_indices,self.proj_valid,self.proj_negative,self.proj_target,self.proj_weights]))
        ix,valid,negative,target,weight=self._projected_cache[key];canonical=p[torch.as_tensor(order,device=p.device)];pp=canonical[ix]
        values=_GroupedCDF.apply(torch.where(negative,1-pp,pp)*valid,target,self.proj_bounds,self.proj_caps)
        loss=(weight*values).sum()
        if not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite projected loss')
        return loss,values
