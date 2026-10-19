"""Complete small-graph truth, exact finite sums, and iid inverse-CDF draws.

No MCMC, role locking or uniform-role-mixture approximation. Probabilities are
FP64 full-enumeration calculations, not symbolic/infinite-precision claims.
"""
import itertools
import numpy as np
from scipy.special import logsumexp
from .contracts import CONFIG,validate_target


class Universe:
    def __init__(self,schema):
        if schema.s>15 or schema.n>6:raise ValueError('enumeration restricted to at most6 nodes/15 slots')
        self.schema=schema;self.codes=np.arange(1<<schema.s,dtype=np.uint32)
        self.bits=((self.codes[:,None]>>np.arange(schema.s))&1).astype(np.uint8)
        inc=np.zeros((schema.s,schema.n*schema.r),np.int64)
        for e,(u,v) in enumerate(schema.slots.T):
            inc[e,u*schema.r+schema.families[e]]+=1;inc[e,v*schema.r+schema.families[e]]+=1
        self.degrees=(self.bits@inc).reshape(-1,schema.n,schema.r)
        adjacency=np.zeros((len(self.codes),schema.n,schema.n),bool)
        for e,(u,v) in enumerate(schema.slots.T):adjacency[:,u,v]=adjacency[:,v,u]=self.bits[:,e]
        reached=np.zeros((len(self.codes),schema.n),bool);reached[:,0]=True
        for _ in range(schema.n-1):reached |= np.any(adjacency & reached[:,:,None],axis=1)
        self.connected=reached.all(axis=1)
        self.triangles=np.zeros(len(self.codes),np.int64)
        for a,b,c in itertools.combinations(range(schema.n),3):self.triangles+=adjacency[:,a,b]&adjacency[:,a,c]&adjacency[:,b,c]
        # Exhaustive type-preserving permutation orbits. Hashes never certify
        # isomorphism here: every orbit member has an explicit permutation.
        choices=[list(itertools.permutations(np.flatnonzero(schema.types==c))) for c in schema.sizes]
        permutations=[]
        for choice in itertools.product(*choices):
            perm=np.arange(schema.n)
            for c,values in zip(schema.sizes,choice):perm[schema.types==c]=values
            permutations.append(perm)
        self.permutations=np.array(permutations)
        lookup=np.full((schema.n,schema.n),-1,int)
        for e,(u,v) in enumerate(schema.slots.T):lookup[u,v]=lookup[v,u]=e
        mapped=lookup[self.permutations[:,schema.slots[0]],self.permutations[:,schema.slots[1]]]
        if (mapped<0).any():raise AssertionError('type-preserving permutation left schema')
        self.topology=np.full(len(self.codes),-1,np.int64)
        for code in self.codes:
            if self.topology[code]>=0:continue
            present=np.flatnonzero(self.bits[code])
            orbit=np.unique(np.sum(np.left_shift(np.uint64(1),mapped[:,present].astype(np.uint64)),axis=1,dtype=np.uint64))
            if (self.topology[orbit]>=0).any():raise AssertionError('permutation orbits overlap')
            self.topology[orbit]=int(orbit.min())
        if (self.topology<0).any():raise AssertionError('incomplete exact isomorphism partition')

    def role_mask(self,target,marginals_only=False):
        validate_target(self.schema,target);keep=np.ones(len(self.codes),bool)
        for c,g in target['groups'].items():
            degree=self.degrees[:,self.schema.types==int(c),:]
            if marginals_only:
                rows=np.repeat(np.asarray(g['rows']),g['counts'],axis=0)
                for r in range(self.schema.r):keep &= np.all(np.sort(degree[:,:,r],axis=1)==np.sort(rows[:,r])[None,:],axis=1)
            else:
                for row,count in zip(g['rows'],g['counts']):keep &= np.sum(np.all(degree==np.array(row),axis=2),axis=1)==count
        return keep

    def relabel_codes(self,permutation):
        s=self.schema;p=np.asarray(permutation)
        if sorted(p.tolist())!=list(range(s.n)) or not np.array_equal(s.types[p],s.types):raise ValueError('type-preserving permutation required')
        lookup=np.full((s.n,s.n),-1,int)
        for e,(u,v) in enumerate(s.slots.T):lookup[u,v]=lookup[v,u]=e
        mapped=lookup[p[s.slots[0]],p[s.slots[1]]]
        codes=self.bits.astype(np.uint64)@np.left_shift(np.uint64(1),mapped.astype(np.uint64))
        return mapped,codes


def normalized(logweights,mask):
    mask=np.asarray(mask,bool)&np.isfinite(logweights)
    if not mask.any():raise ValueError('empty feasible model support; conditional distribution undefined')
    z=float(logsumexp(logweights[mask]));p=np.zeros(len(mask));p[mask]=np.exp(logweights[mask]-z)
    error=abs(float(p.sum())-1.)
    if error>CONFIG['normalization_atol']:raise FloatingPointError('full-enumeration normalization failed')
    return p,dict(log_partition=z,normalization_error=error,mathematical_support_size=int(mask.sum()),
        numerical_positive_count=int((p>0).sum()),numerical_underflow_count=int((mask&(p==0)).sum()))


def distributions(universe,target,law):
    if len(law.logits)!=universe.schema.s:raise ValueError('law/schema mismatch')
    roles=universe.role_mask(target);allowed=law.allowed(universe.bits);feasible=roles&allowed
    logq=law.log_prob(universe.bits)
    raw,raw_info=normalized(logq,allowed)
    conditional,cond_info=normalized(logq,feasible)
    warm,warm_info=normalized(logq/2,feasible)
    uniform,uniform_info=normalized(np.zeros_like(logq),feasible)
    best=float(logq[feasible].max());representative=int(np.flatnonzero(feasible&(logq==best))[0])
    map_p=np.zeros(len(logq));map_p[representative]=1.
    info=dict(C_size=int(roles.sum()),admissible_C_size=int(feasible.sum()),
        q_C=float(np.exp(cond_info['log_partition'])),log_q_C=cond_info['log_partition'],
        MAP=dict(scope='anonymous GLOBAL optimum by complete enumeration ONLY',representative_code=representative,
            exact_FP64_tied_maximizers=int((feasible&(logq==best)).sum()),
            within_tie_tolerance_count=int((feasible&(logq>=best-CONFIG['enumeration_MAP_tie_atol'])).sum()),
            tie_rule='smallest graph code among exact FP64 maximizers; not an equivariant tie rule'),
        distributions=dict(independent=raw_info,uniform_feasible=uniform_info,conditional_T1=cond_info,conditional_T2=warm_info))
    return dict(independent=raw,MAP=map_p,uniform_feasible=uniform,conditional_T1=conditional,conditional_T2=warm),info,logq,roles


def draw(probability,uniforms):
    cdf=np.cumsum(probability);error=abs(float(cdf[-1])-1)
    if error>CONFIG['normalization_atol']:raise FloatingPointError('cannot sample unnormalized distribution')
    cdf[-1]=1. # terminal cumulative roundoff only, no epsilon edge support
    return np.searchsorted(cdf,uniforms,side='right')


def metrics(universe,p,logq,role_mask,draws,mathematical_support):
    positive=p>0;entropy=float(-np.sum(p[positive]*np.log(p[positive])))
    marginals=p@universe.bits
    labels,mapping=np.unique(universe.topology,return_inverse=True)
    masses=np.bincount(mapping,weights=p,minlength=len(labels));top=masses>0
    topology_entropy=float(-np.sum(masses[top]*np.log(masses[top])))
    # Exact expectation of distinct labelled graphs in4096 iid draws, distinct
    # from the observed count. P=1 is handled as a limit, never log(0)*0.
    expected_distinct=np.ones(len(p));take=p<1
    expected_distinct[take]=-np.expm1(len(draws)*np.log1p(-p[take]))
    return dict(schema_simple_undirected_rate=1.,exact_joint_role_rate=float(p@role_mask),
        mathematical_support_size=int(mathematical_support),numerically_positive_support=int(positive.sum()),
        entropy_nats=entropy,effective_graph_count_exp_entropy=float(np.exp(entropy)),
        inverse_simpson_graph_count=float(1/(p@p)),maximum_single_graph_probability=float(p.max()),
        sampled_distinct_labelled_graphs=int(len(np.unique(draws))),expected_distinct_labelled_graphs=float(expected_distinct.sum()),
        expected_two_graph_edge_symmetric_difference=float(2*np.sum(marginals*(1-marginals))),
        connected_rate=float(p@universe.connected),mean_triangles=float(p@universe.triangles),
        mean_original_log_q_nats=float(p[positive]@logq[positive]),
        mean_original_log_q_per_slot=float(p[positive]@logq[positive]/max(1,universe.schema.s)),
        exact_type_preserving_topology_classes=int(top.sum()),topology_entropy_nats=topology_entropy,
        effective_topology_classes=float(np.exp(topology_entropy)),maximum_topology_mass=float(masses.max()),
        sampled_distinct_topologies=int(len(np.unique(universe.topology[draws]))),
        observed_joint_role_passes=int(role_mask[draws].sum()),
        topology_frequency=[dict(canonical_graph_code=int(c),probability=float(w),draw_count=int((universe.topology[draws]==c).sum())) for c,w in zip(labels[top],masses[top])])


def uniform_assignment_counterexample(universe,conditional,logq,role_mask):
    """Uniform role assignment then q|assignment weights each group incorrectly."""
    codes=np.flatnonzero(role_mask&np.isfinite(logq))
    matrices=universe.degrees[codes].reshape(len(codes),-1)
    _,group=np.unique(matrices,axis=0,return_inverse=True);k=int(group.max())+1
    mixture=np.zeros(len(logq));partitions=[]
    for g in range(k):
        members=codes[group==g];z=float(logsumexp(logq[members]));partitions.append(z)
        mixture[members]=np.exp(logq[members]-z)/k
    return dict(distinct_complete_role_assignments=k,min_log_assignment_partition=min(partitions),
        max_log_assignment_partition=max(partitions),
        TV_uniform_assignment_mixture_vs_true_conditional=float(abs(mixture-conditional).sum()/2),
        meaning='uniform role assignment is not q conditional unless all assignment partition functions are equal')
