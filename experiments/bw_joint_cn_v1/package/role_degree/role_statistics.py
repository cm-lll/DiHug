"""Anonymous JOINT degree statistics only. No model, sampler or guidance update.

Target extraction may read the permitted training graph. Its exported payload
contains only canonical degree-vector multiplicities by type, never node IDs,
assignments or edges. Probability statistics require a complete legal slot table.
"""
import numpy as np


def validate(types, edges, families, pairs):
    types=np.asarray(types,dtype=np.int64)
    edges=np.asarray(edges,dtype=np.int64)
    families=np.asarray(families,dtype=np.int64)
    pairs=np.asarray(pairs,dtype=np.int64)
    if types.ndim!=1 or len(types)==0 or (types<0).any():raise ValueError("invalid types")
    if pairs.ndim!=2 or pairs.shape[1]!=2 or (pairs<0).any() or (pairs[:,0]>pairs[:,1]).any():
        raise ValueError("invalid family pairs")
    if len(np.unique(pairs,axis=0))!=len(pairs):raise ValueError("duplicate family")
    if edges.shape!=(2,len(families)) or families.ndim!=1:raise ValueError("edge/family shape")
    if ((edges<0)|(edges>=len(types))).any() or (edges[0]>=edges[1]).any():
        raise ValueError("canonical physical edges required")
    if ((families<0)|(families>=len(pairs))).any():raise ValueError("family range")
    if len(np.unique(edges[0]*len(types)+edges[1]))!=len(families):raise ValueError("duplicate slot")
    if not np.array_equal(np.sort(types[edges],axis=0).T,pairs[families]):raise ValueError("schema mismatch")
    return types,edges,families,pairs


def joint_profile(types,edges,families,pairs):
    types,edges,families,pairs=validate(types,edges,families,pairs)
    degrees=np.zeros((len(types),len(pairs)),dtype=np.int64)
    for endpoint in edges:np.add.at(degrees,(endpoint,families),1)
    groups={}
    for ty in np.unique(types):
        rows,count=np.unique(degrees[types==ty],axis=0,return_counts=True)
        groups[str(ty)]=dict(nodes=int(count.sum()),rows=rows.tolist(),counts=count.tolist())
    return dict(format="anonymous-joint-relation-degree-v1",family_pairs=pairs.tolist(),groups=groups)


def target_cf(profile,omega):
    """Exact characteristic function of the exported empirical joint law."""
    omega=np.asarray(omega,dtype=np.float64)
    if omega.ndim!=2 or omega.shape[1]!=len(profile["family_pairs"]) or not np.isfinite(omega).all():
        raise ValueError("frequency dimensions/nonfinite")
    result={}
    for ty,g in profile["groups"].items():
        rows=np.asarray(g["rows"],dtype=np.float64);count=np.asarray(g["counts"],dtype=np.float64)
        if count.sum()!=g["nodes"] or (count<=0).any():raise ValueError("target counts")
        result[ty]=(count[:,None]*np.exp(1j*(rows@omega.T))).sum(0)/g["nodes"]
    return result


def independent_degree_cf(types,slots,families,probabilities,pairs,omega):
    """Exact E[empirical CF], conditional on independent Bernoulli slot p.

    Not CF(expected degree); not E[squared empirical distance]. No sampling or
    autograd. Frequency loop avoids constructing S x K or N x N x K arrays.
    Production-guidance gradients/line search are intentionally NOT implemented.
    """
    types,slots,families,pairs=validate(types,slots,families,pairs)
    expected=0
    for a,b in pairs:
        na=int((types==a).sum());nb=int((types==b).sum())
        expected+=na*(na-1)//2 if a==b else na*nb
    if len(families)!=expected:raise ValueError("complete legal slot table required")
    p=np.asarray(probabilities,dtype=np.float64);omega=np.asarray(omega,dtype=np.float64)
    if p.shape!=families.shape or not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError("probability")
    if omega.ndim!=2 or omega.shape[1]!=len(pairs) or not np.isfinite(omega).all():raise ValueError("frequency")
    phi=np.ones((len(types),len(omega)),dtype=np.complex128)
    for k in range(len(omega)):
        factor=1+p*(np.exp(1j*omega[k,families])-1)
        for endpoint in slots:np.multiply.at(phi[:,k],endpoint,factor)
    return {str(ty):phi[types==ty].mean(0) for ty in np.unique(types)}


def cf_discrepancy(first,second):
    if set(first)!=set(second) or not first:raise ValueError("type populations differ")
    values=[]
    for ty in first:
        a,b=np.asarray(first[ty]),np.asarray(second[ty])
        if a.shape!=b.shape or a.size==0:raise ValueError("frequency counts differ")
        values.append(float(np.mean(np.abs(a-b)**2)))
    return float(np.mean(values))
