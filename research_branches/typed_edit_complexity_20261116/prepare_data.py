"""Reference-only data and feasibility audit before any new model outcome."""
from common import *
import zipfile,io
from collections import Counter
from scipy import sparse,io as sio
from dihug.core.target_feasibility import validate_role_target

def source_freebase():
    p=RUN/'Freebase.zip';z=zipfile.ZipFile(p)
    text=z.read('Freebase/info.dat').decode();lines=text.splitlines();start=lines.index('TYPE\tMEANING')+1
    names=[]
    while lines[start].strip():names.append(lines[start].split()[-1]);start+=1
    nodes=[line.decode().rstrip('\n').split('\t') for line in z.open('Freebase/node.dat')]
    ids=np.array([int(x[0]) for x in nodes]);raw_types=np.array([int(x[2]) for x in nodes])
    assert np.array_equal(np.sort(ids),np.arange(len(ids))), 'IDs must be unique and contiguous; row order is irrelevant'
    types=np.empty(len(ids),dtype=int);types[ids]=raw_types
    edges=np.loadtxt(io.BytesIO(z.read('Freebase/link.dat')),dtype=float)
    assert np.isfinite(edges).all() and np.all(edges[:,3]==1)
    families={};endpoints={};rawcounts={}
    for rel in sorted(set(edges[:,2].astype(int))):
        uv=edges[edges[:,2]==rel,:2].astype(np.int64);pairs=set(map(tuple,np.sort(types[uv],axis=1)))
        assert len(pairs)==1;endpoints[f'r{rel:02}']=next(iter(pairs));rawcounts[f'r{rel:02}']=len(uv)
        uv=np.sort(uv,axis=1);families[f'r{rel:02}']=np.unique(uv[uv[:,0]!=uv[:,1]],axis=0)
    assert len(names)==8 and len(families)==36 and len(set(endpoints.values()))==36
    return types,names,families,endpoints,{'archive':str(p),'sha256':sha(p),'source':'HGB NC Freebase; PyG published file ID 1vw-uqbroJZfFsWpriC1CWbtHCJMGdWJ7',
        'official_repository_commit':'ca6fd5bb0c1ca32e63b132c8bfe8f11a4a6629fe','raw_rows':len(edges),'raw_family_counts':rawcounts,
        'transformation':'binary undirected, remove self-loops, collapse reciprocal/duplicate rows; preserve all 36 relation identities',
        'info':text,'classification_labels_used':False,'old_download_response':'Freebase.download is HTML, retained as failed endpoint evidence'}

def source_acm():
    p=ROOT/'data/external/new_hetero_20261112/HAN_ACM.mat';assert sha(p)=='0ccd838e545e8f16e3dc84356da2f51dfd2290c32e37a784e374dd510c76578d'
    m=sio.loadmat(p);names=['Paper','Author','Term','Venue','Affiliation','Subject','Conference']
    pairs={'PvsA':(0,1),'PvsT':(0,2),'PvsV':(0,3),'AvsF':(1,4),'PvsL':(0,5),'PvsC':(0,6),'VvsC':(3,6),'PvsP':(0,0)}
    counts=[12499,17431,1903,196,1804,73,14];offset=np.cumsum([0]+counts);types=np.repeat(np.arange(7),counts);families={};rawcounts={}
    for key,(a,b) in pairs.items():
        q=m[key].tocoo();assert q.shape==(counts[a],counts[b]);assert np.all(q.data==1)
        uv=np.sort(np.column_stack([q.row+offset[a],q.col+offset[b]]),axis=1);rawcounts[key]=len(uv)
        families[key]=np.unique(uv[uv[:,0]!=uv[:,1]],axis=0)
    return types,names,families,pairs,{'source':str(p),'sha256':sha(p),'repository':'https://github.com/Jhy1993/HAN',
        'source_commit':'71bac29a07fb8fab908d50a806a7bc38aa6c6611','raw_family_counts':rawcounts,
        'transformation':'all eight direct relation matrices; binary undirected; self-loop removal and reciprocal deduplication',
        'not_used':'later 192-node prepared views, models or outcomes; classification labels; projected metapaths'}

def choose(types,families,n=2048):
    all_edges=np.concatenate(list(families.values()));rows=np.concatenate([all_edges[:,0],all_edges[:,1]]);cols=np.concatenate([all_edges[:,1],all_edges[:,0]])
    adj=sparse.csr_matrix((np.ones(len(rows),dtype=np.int8),(rows,cols)),shape=(len(types),len(types)));adj.sum_duplicates();adj.sort_indices()
    # Equal-type capped inventory quotas, fixed independently of model outcomes.
    counts=np.bincount(types);quota=np.zeros(len(counts),dtype=int)
    for _ in range(n):
        eligible=np.flatnonzero(quota<counts);t=min(eligible,key=lambda t:(quota[t],t));quota[t]+=1
    root=int(np.flatnonzero((types==0)&(np.diff(adj.indptr)>0))[0]);rng=np.random.default_rng(2026111602)
    seen=set();front=[set() for _ in counts];front[int(types[root])].add(root);actual=np.zeros(len(counts),int)
    while len(seen)<n:
        eligible=[t for t in range(len(counts)) if front[t]]
        if not eligible:raise ValueError('fixed root component too small; no automatic replacement')
        t=min(eligible,key=lambda t:(actual[t]/max(1,quota[t]),t));options=sorted(front[t]);u=options[int(rng.integers(len(options)))];front[t].remove(u)
        assert u not in seen;seen.add(u);actual[t]+=1
        for v in adj.indices[adj.indptr[u]:adj.indptr[u+1]]:
            if int(v) not in seen:front[int(types[v])].add(int(v))
    return np.array(sorted(seen)),{'algorithm':'fixed-root type-balanced frontier sampling; exact induced edges','seed':2026111602,'root':root,'quota':quota.tolist(),'actual':actual.tolist()}

def save(ds):
    tic=time.time();out=RUN/'data'/ds;out.mkdir(parents=True,exist_ok=False)
    if ds in ['imdb','pubmed']:
        p=ROOT/f'results/paper_revision_pr3b/pr3b_20261008_stage2_attribution_v1/private_reference/{ds}.json';g=restore_graph(read(p));source={'path':str(p),'sha256':sha(p),'selection':'exact historical reference, unchanged'}
    else:
        types,names,edges,endpoints,source=(source_freebase() if ds=='freebase' else source_acm())
        ids,selection=choose(types,edges);remap=np.full(len(types),-1,dtype=int);remap[ids]=np.arange(len(ids));ts=types[ids]
        nodes={t:tuple(map(int,np.flatnonzero(ts==t))) for t in range(len(names))}
        specs={f:FamilySpec(f,int(a),int(b),nodes[a],nodes[b]) for f,(a,b) in endpoints.items()}
        selected={f:uv[(remap[uv]>=0).all(1)] for f,uv in edges.items()}
        g=TypedGraph(ts,names,specs,{f:remap[uv].tolist() for f,uv in selected.items()});source['full_nodes']=len(types);source['full_edges']=sum(map(len,edges.values()));source['selection']=selection
        put(out/'node_map.json',{'original_ids':ids.tolist(),'types':ts.tolist()})
        # Independently map saved edges back to source, exact induced membership.
        for f,uv in edges.items():
            expected={tuple(x) for x in uv if remap[x[0]]>=0 and remap[x[1]]>=0}
            restored={tuple(sorted((int(ids[u]),int(ids[v])))) for u,v in g.family_edges[f]};assert expected==restored
    g.validate();g.validate_disjoint_family_domains();put(out/'graph.json',graph_payload(g));prior=MarginalGraphPrior.from_training_graph(g);role=ExchangeableRoleDegreePrior.from_training_graph(g);role.save(out/'role.json')
    diagnostics=[]
    for seed in range(100,108):
        initial=FamilyCountCondition(prior,role.family_counts(prior.family_endpoints)).sample_graph(seed+71,mode='binomial');target=role.sample_assignment_for_graph(initial,seed+91919)
        d=np.column_stack([initial.family_degrees(f) for f in sorted(initial.family_specs)]);distance=int(np.abs(d-target).sum());lower=max(0,(distance-math.isqrt((target.size*11*g.num_nodes)//2))//4)
        feasible=validate_role_target(initial,target);put(out/'starts'/f'seed{seed}'/'initial.json',graph_payload(initial));put(out/'starts'/f'seed{seed}'/'target.json',target.tolist())
        diagnostics.append({'seed':seed,'L1':distance,'necessary_lower':lower,'feasibility':feasible})
    # Common rule, reference/start-only: four times optimistic lower bound, at least 10k.
    budget=max(10000,int(math.ceil(4*max(x['necessary_lower'] for x in diagnostics)/1000))*1000)
    width=mixed_edit_feature_width('typed_closure_compact_v2',len(g.family_specs),len(g.type_names));params={a:sum(p.numel() for p in model(width,a,0).parameters()) for a in ['M','F','W']}
    report={'dataset':ds,'source':source,'nodes':g.num_nodes,'edges':g.num_edges,'types':g.type_names,
        'type_counts':np.bincount(g.node_types).tolist(),'families':{f:len(es) for f,es in g.family_edges.items()},
        'endpoints':{f:[s.src_type,s.dst_type] for f,s in g.family_specs.items()},'starts':diagnostics,'steps':budget,'horizon':2*budget,
        'feature_width':width,'parameters':params,'W_width':architecture(width,'W')[0],
        'graph_sha256':sha(out/'graph.json'),'seconds':time.time()-tic,'passed':budget<=50000,
        'note':'graphicality is reported, not used to reject targets; zero-count families retained; no model outcomes used'}
    put(out/'audit.json',report);print(json.dumps({k:v for k,v in report.items() if k not in ['source','starts']}),flush=True)
    assert report['passed'],'necessary budget exceeds fixed 50k engineering cap; retain reference and stop before training'
if __name__=='__main__':
    setup()
    for ds in ['imdb','pubmed','freebase','acm']:save(ds)
