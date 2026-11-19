"""CPU-only descriptor audit. No denoiser, optimizer, or new real graph draws."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[k] = '1'
import csv, hashlib, itertools, json, resource, signal, time
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'results/hard_clean_zero_safe_20261110'
OUT = ROOT/'results'/('typed_score_cpu_20261110'+os.environ.get('AUDIT_RUN_SUFFIX',''))

def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def table(path, rows):
    if not rows: return
    with Path(path).open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

class Descriptor:
    """Relations are unique per legal unordered pair in the bound data contract."""
    def __init__(self, types, query, families, reference):
        self.types = np.asarray(types, dtype=int)
        self.q = np.asarray(query, dtype=int)
        self.f = np.asarray(families, dtype=int)
        self.n = len(types); self.r = int(self.f.max())+1
        self.type_ids = np.unique(self.types)
        assert self.q.shape == (2, len(self.f))
        assert np.all(self.q[0] < self.q[1])
        assert len(set(map(tuple, self.q.T))) == len(self.f)
        self.support = np.zeros((self.n, self.n), dtype=bool)
        self.support[tuple(self.q)] = True; self.support |= self.support.T
        self.fm = np.full((self.n, self.n), -1, dtype=int)
        self.fm[tuple(self.q)] = self.f; self.fm[tuple(self.q[::-1])] = self.f
        self.ref = self.features(reference)
        inventory = self.features(np.ones(len(self.f), dtype=np.uint8))['local']
        self.masks = {}; self.scales = {}; self.component_masks = {}
        width = self.r**2
        for typ in self.type_ids:
            ix = self.types == typ
            # At full inventory, every closable wedge is closed. Open versions
            # of the SAME feasible channels must remain in the description.
            possible = inventory[ix].sum(0) > 0
            possible[width:2*width] = possible[:width]
            self.masks[int(typ)] = possible
            ref = self.ref['local'][ix][:, possible]
            self.scales[int(typ)] = np.maximum(1., np.sqrt(np.mean(ref**2, axis=0)))
            self.component_masks[int(typ)] = np.array([
                np.sum(self.types == a) - int(a == typ) > 0 for a in self.type_ids])

    def features(self, bits):
        bits = np.asarray(bits, dtype=bool)
        assert bits.shape == (len(self.f),)
        a = np.zeros((self.n, self.n), dtype=bool)
        i, j = self.q[:, bits]; a[i,j] = True; a[j,i] = True
        degree = np.zeros((self.n, self.r), dtype=float)
        np.add.at(degree, (i, self.f[bits]), 1); np.add.at(degree, (j, self.f[bits]), 1)
        width = self.r**2
        local = np.zeros((self.n, 3*width), dtype=float)
        for u in range(self.n):
            nbr = np.flatnonzero(a[u]); ii, jj = np.triu_indices(len(nbr), 1)
            v, w = nbr[ii], nbr[jj]
            left, right = self.fm[u,v], self.fm[u,w]
            key = np.minimum(left,right)*self.r + np.maximum(left,right)
            closed = a[v,w]; eligible = self.support[v,w]
            mode = np.where(closed, 0, np.where(eligible, 1, 2))
            local[u] = np.bincount(mode*width+key, minlength=3*width)
        nc, labels = connected_components(csr_matrix(a), directed=False)
        sizes = np.bincount(labels)
        hist = np.array([[np.sum((labels == c)&(self.types == typ)) for typ in self.type_ids] for c in range(nc)])
        comp = hist[labels].astype(float)
        for col, typ in enumerate(self.type_ids):
            own = (self.types == typ).astype(int)
            comp[:,col] = (comp[:,col]-own)/np.maximum(1, np.sum(self.types == typ)-own)
        return dict(degree=degree, local=local, component=comp,
                    edges=int(bits.sum()), components=int(nc), LCC=float(sizes.max()/self.n),
                    isolates=int(np.sum(a.sum(1) == 0)), triangles=float(local[:,:width].sum()/3))

    @staticmethod
    def match(cost):
        i,j = linear_sum_assignment(cost)
        return float(cost[i,j].mean())

    def score(self, bits):
        ft = self.features(bits); pertype = []
        for typ in self.type_ids:
            ix = self.types == typ; mask = self.masks[int(typ)]
            a = cdist(ft['degree'][ix], self.ref['degree'][ix], 'cityblock')/max(1,self.n-1)
            if mask.any():
                scale = self.scales[int(typ)]
                b = cdist(ft['local'][ix][:,mask]/scale, self.ref['local'][ix][:,mask]/scale, 'cityblock')/mask.sum()
            else: b = np.zeros_like(a)
            cm = self.component_masks[int(typ)]
            c = cdist(ft['component'][ix][:,cm], self.ref['component'][ix][:,cm], 'cityblock')/max(1, cm.sum())
            jd, loc, joint, conn = map(self.match, (a,b,a+b,c))
            assert joint+1e-12 >= jd+loc
            pertype.append(dict(node_type=int(typ), nodes=int(ix.sum()), J=jd, L=loc, JL=joint,
                                assignment_gap=max(0.,joint-jd-loc), C=conn,
                                local_channels=int(mask.sum()), max_degree=float(ft['degree'][ix].sum(1).max()),
                                zero_degree_fraction=float(np.mean(ft['degree'][ix].sum(1)==0))))
        result = {name:float(np.mean([p[name] for p in pertype])) for name in ('J','L','JL','assignment_gap','C')}
        result.update({k:ft[k] for k in ('edges','components','LCC','isolates','triangles')})
        return result, pertype

def graph(n, edges, types=None, support=None):
    types = np.zeros(n,dtype=int) if types is None else np.array(types)
    q = np.array(list(itertools.combinations(range(n),2)) if support is None else support, dtype=int).T
    pairs = [tuple(sorted((int(types[i]),int(types[j])))) for i,j in q.T]
    keys = sorted(set(pairs)); f = np.array([keys.index(p) for p in pairs])
    edge_set = set(tuple(sorted(e)) for e in edges)
    return types,q,f,np.array([tuple(e) in edge_set for e in q.T],dtype=np.uint8)

def tests():
    count = 0
    # Independent triple enumeration on all 64 four-node graphs, two schemas.
    for restricted in (False,True):
        typ,q,f,_ = graph(4,[],[0,1,0,1], [(0,1),(0,3),(1,2),(2,3)] if restricted else None)
        d = Descriptor(typ,q,f,np.zeros(len(f),dtype=np.uint8))
        for bit_tuple in itertools.product((0,1), repeat=len(f)):
            b = np.array(bit_tuple); got = d.features(b); expected = np.zeros_like(got['local'])
            present = set(tuple(e) for e in q[:,b.astype(bool)].T)
            legal = {tuple(e):int(r) for e,r in zip(q.T,f)}
            for u in range(4):
                for v,w in itertools.combinations([i for i in range(4) if i != u],2):
                    uv,uw,vw = tuple(sorted((u,v))),tuple(sorted((u,w))),tuple(sorted((v,w)))
                    if uv not in present or uw not in present: continue
                    lo,hi = sorted((legal[uv],legal[uw]))
                    mode = 0 if vw in present else (1 if vw in legal else 2)
                    expected[u,mode*d.r**2+lo*d.r+hi] += 1
            np.testing.assert_array_equal(got['local'],expected); count+=1
    # Typed bipartite components: identical degree, triangles and component count.
    cycle = lambda nodes: [(nodes[i],nodes[(i+1)%len(nodes)]) for i in range(len(nodes))]
    a = cycle(list(range(4)))+cycle(list(range(4,12)))
    b = cycle(list(range(6)))+cycle(list(range(6,12)))
    types = np.arange(12)%2; support=[(i,j) for i in range(12) for j in range(i+1,12) if types[i]!=types[j]]
    typ,q,f,ref=graph(12,a,types,support); _,_,_,alt=graph(12,b,types,support)
    d=Descriptor(typ,q,f,ref); component_case=d.score(alt)[0]
    assert component_case['J']==component_case['L']==0 and component_case['C']>0
    assert d.features(ref)['components']==component_case['components']==2
    # Explicit remaining blind spot: connected triangle-free cubic graphs.
    cube=[(i,i^x) for i in range(8) for x in (1,2,4) if i<(i^x)]
    mobius=cycle(list(range(8)))+[(i,i+4) for i in range(4)]
    typ,q,f,ref=graph(8,cube); _,_,_,alt=graph(8,mobius)
    d=Descriptor(typ,q,f,ref); blind=d.score(alt)[0]
    assert all(abs(blind[k])<1e-12 for k in ('J','L','JL','C'))
    def squares(edges):
        es=set(tuple(sorted(e)) for e in edges); total=0
        for nodes in itertools.combinations(range(8),4):
            for rest in itertools.permutations(nodes[1:]):
                order=(nodes[0],)+rest
                total += all(tuple(sorted((order[i],order[(i+1)%4]))) in es for i in range(4))
        return total//2
    assert squares(cube)!=squares(mobius)
    # Fixed exhaustive space, deterministic first witness. No tuning on real data.
    q6=np.array(list(itertools.combinations(range(6),2))).T
    seen={}; witness=None; joint_witness=None; seen_joint={}
    for code in range(1<<15):
        bits=((code>>np.arange(15))&1).astype(np.uint8)
        a=np.zeros((6,6),dtype=np.int64); a[tuple(q6)]=bits; a+=a.T
        deg=a.sum(1); tri=(a@a*a).sum(1)//2
        key=(tuple(sorted(deg)),int(tri.sum()))
        local=tuple(sorted(tri)); paired=tuple(sorted(zip(deg.tolist(),tri.tolist())))
        if witness is None and key in seen and seen[key][1]!=local:
            witness=(seen[key][0],code)
        seen.setdefault(key,(code,local))
        jk=(key[0],local)
        if joint_witness is None and jk in seen_joint and seen_joint[jk][1]!=paired:
            joint_witness=(seen_joint[jk][0],code)
        seen_joint.setdefault(jk,(code,paired))
    assert witness is not None
    cases=[]
    for label,pair in [('same_degrees_triangle_total',witness),('same_degree_triangle_marginals',joint_witness)]:
        if pair is None: cases.append(dict(name=label,found=False)); continue
        ref=((pair[0]>>np.arange(15))&1).astype(np.uint8); alt=((pair[1]>>np.arange(15))&1).astype(np.uint8)
        d=Descriptor(np.zeros(6,dtype=int),q6,np.zeros(15,dtype=int),ref)
        result=d.score(alt)[0]; assert result['J']==0 and result['JL']>0
        cases.append(dict(name=label,found=True,codes=list(pair),scores=result,
                          reference_edges=q6[:,ref.astype(bool)].T.tolist(),comparison_edges=q6[:,alt.astype(bool)].T.tolist()))
    return dict(passed=True,independent_feature_graphs=count,exhaustive_six_node_graphs=32768,
                local_cases=cases,typed_component_case=component_case,
                blind_spot=dict(scores=blind,cube_4cycles=squares(cube),mobius_4cycles=squares(mobius)))

def main():
    start=time.time(); OUT.mkdir(exist_ok=False)
    def timeout(*_): raise TimeoutError('300 second CPU bound')
    signal.signal(signal.SIGALRM,timeout); signal.alarm(int(os.environ.get('AUDIT_WALL_SECONDS','300')))
    def terminated(*_): raise RuntimeError('watchdog stop: wall or resident-memory bound')
    signal.signal(signal.SIGTERM,terminated)
    receipt=dict(start_epoch=start,GPU_hours=0,model_calls=0,updates=0,new_real_graph_draws=0)
    rows=[]; pertype=[]; batches=[]
    try:
        original=json.loads((SOURCE/'protocol.json').read_text()); tasks=original['tasks']
        bindings={str(Path(__file__)):sha(__file__),str(SOURCE/'protocol.json'):sha(SOURCE/'protocol.json')}
        for row in tasks:
            paths=[Path(row['evaluation_file']),Path(row['data']),SOURCE/row['task']/'result.json',
                   ROOT/'results/hard_clean_pilot_v2_20261110'/row['task']/'score_definition.json']
            paths+=list((SOURCE/row['task']).glob('*.npz'))
            for p in paths: bindings[str(p)]=sha(p)
        dump(OUT/'protocol.json',dict(bindings=bindings,CPU_threads=1,wall_seconds=300,memory_GiB=2,GPU=0,
            full_saved_panel=dict(tasks=4,batches=52,graphs=1600),
            legal_support='actual canonical query inventory, excludes fixed-zero slots',
            local_channels='center type + unordered incident relation pair; closed/eligible-open/unclosable',
            local_scale='frozen reference per-type RMS, floor 1; feasible inventory channels retained even if reference zero',
            degree='unchanged Hungarian mean L1 joint relation degree divided by N-1, types equal weight',
            component='per-node reachable other nodes of each type / available other-node inventory; type-wise Hungarian mean L1',
            joint='Hungarian on degree cost + local cost, diagnostic only, not adopted loss or justified gradient balancing',
            controlled='all 4-node graphs of full/two-type bipartite schema; lexicographic 6-node witnesses; typed C4+C8 vs C6+C6; cube vs Mobius blind spot',
            no_threshold_tuning=True,no_training=True,no_score_combination_adoption=True))
        prior_tests=os.environ.get('AUDIT_REUSE_TESTS')
        if prior_tests:
            dump(OUT/'cpu_tests.json',json.loads(Path(prior_tests).read_text()))
            dump(OUT/'test_reuse.json',dict(path=prior_tests,sha256=sha(prior_tests),reason='only memory supervision changed; scientific code unchanged'))
        else: dump(OUT/'cpu_tests.json',tests())
        import torch
        torch.set_num_threads(1)
        for task in tasks:
            inputs=torch.load(task['evaluation_file'],map_location='cpu',weights_only=False)['inputs']
            inp=inputs[0]; order=np.argsort(inp['ids']); q=inp['query'][:,order]; f=inp['f'][order]
            ref=inp['labels'][order]; ids=inp['ids'][order]
            data=np.load(task['data']); types=data['types']
            d=Descriptor(types,q,f,ref)
            definition=json.loads((ROOT/'results/hard_clean_pilot_v2_20261110'/task['task']/'score_definition.json').read_text())
            np.testing.assert_array_equal(d.ref['degree'],np.array(definition['reference']['degree'])[:,:d.r])
            assert d.ref['components']==definition['reference']['components']
            base=d.score(ref)[0]; assert all(base[k]==0 for k in ('J','L','JL','C'))
            # Type-preserving permutation changes identities, not anonymous scores.
            rng=np.random.default_rng(2026111002); perm=np.arange(len(types))
            for typ in d.type_ids:
                ix=np.flatnonzero(types==typ); perm[ix]=rng.permutation(ix)
            edges=set(tuple(sorted(perm[e])) for e in q[:,ref.astype(bool)].T)
            relabel=np.array([tuple(e) in edges for e in q.T])
            rr=d.score(relabel)[0]; assert max(rr[k] for k in ('J','L','JL','C'))<1e-12
            channels=[]
            for typ in d.type_ids:
                for col,scale in zip(np.flatnonzero(d.masks[int(typ)]),d.scales[int(typ)]):
                    mode,key=divmod(int(col),d.r**2); r1,r2=divmod(key,d.r)
                    channels.append(dict(node_type=int(typ),mode=['closed','eligible_open','unclosable'][mode],relation_left=r1,relation_right=r2,reference_scale=float(scale)))
            dump(OUT/f'{task["task"]}_features.json',dict(channels=channels,reference={k:base[k] for k in ('edges','components','LCC','isolates','triangles')},permutation_passed=True))
            info=json.loads((SOURCE/task['task']/'result.json').read_text())
            pending=[('failure',info['t'],SOURCE/task['task']/'failed_batch_replay.npz')]
            pending += [(state,t,SOURCE/task['task']/f'{state}_t{t}.npz') for state in ('before','after') for t in original['times']]
            for state,t,path in pending:
                saved=np.load(path); np.testing.assert_array_equal(saved['ids'],ids)
                batch=[]
                for k,bits in enumerate(saved['bits']):
                    value,detail=d.score(bits)
                    np.testing.assert_allclose(value['J'],saved['costs'][k,1],rtol=1e-12,atol=1e-14)
                    meta=dict(task=task['task'],dataset=task['dataset'],seed=task['seed'],state=state,t=t,sample=k)
                    rows.append(dict(**meta,**value,old_component=float(saved['costs'][k,4]),old_triangle_present=float(saved['costs'][k,2]),old_triangle_absent=float(saved['costs'][k,3])))
                    for item in detail: pertype.append(dict(**meta,**item))
                    batch.append(value)
                br=dict(task=task['task'],dataset=task['dataset'],seed=task['seed'],state=state,t=t,samples=len(batch))
                bits=saved['bits'].astype(float); p=saved['probability']; K=len(bits)
                for name in ('J','L','JL','assignment_gap','C','edges','components','LCC','isolates','triangles'):
                    c=np.array([v[name] for v in batch]); br[name]=float(c.mean()); br[name+'_std']=float(c.std(ddof=1))
                    if name in ('L','JL','C'):
                        centered=c-c[0]; adv=(K*centered-centered.sum())/(K-1)
                        br[name+'_unique']=int(len(np.unique(c)))
                        br[name+'_LOO_logit_norm']=float(np.linalg.norm(adv@(bits-p)/K))
                batches.append(br)
                table(OUT/'batches.csv',batches)
                dump(OUT/'progress.json',dict(graphs=len(rows),batches=len(batches),seconds=time.time()-start))
            print(json.dumps(dict(task=task['task'],graphs=len(rows),seconds=time.time()-start)),flush=True)
        for p,h in bindings.items(): assert sha(p)==h
        receipt.update(status='complete',unchanged_bindings=len(bindings))
    except BaseException as exc:
        import traceback
        receipt.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
        raise
    finally:
        table(OUT/'graphs.csv',rows); table(OUT/'types.csv',pertype); table(OUT/'batches.csv',batches)
        receipt.update(end_epoch=time.time(),seconds=time.time()-start,graphs=len(rows),batches=len(batches),peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        dump(OUT/'receipt.json',receipt); print(json.dumps(receipt),flush=True)

if __name__ == '__main__': main()
