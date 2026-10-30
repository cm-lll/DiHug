"""Load author definitions without executing their command-line entry point."""
import ast
import copy
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

ROOT = Path('/data2/lyh/gnn_project/DiHug')
UP = ROOT.parent/'DiHug_paper_revision/results/r4_hgen_motif_20261015_v1/source/vendor/hgen_upstream'
DATA = ROOT.parent/'HGEN/downloads/official_data_20260716'
REF = ROOT/'results/paper_revision_pr3b/pr3b_20261008_stage2_attribution_v1/private_reference'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    tmp.replace(p)

def seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def rng():
    return dict(py=random.getstate(), np=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])

def restore(r):
    random.setstate(r['py']); np.random.set_state(r['np']); torch.set_rng_state(r['torch'])
    if r['cuda']: torch.cuda.set_rng_state_all(r['cuda'])

def args(dataset, epochs=5):
    return SimpleNamespace(batch_size=32, noise_dim=64, noise_type='Uniform', hidden_units=128,
        num_G_layer=5, max_path_len=4, W_down_generator_size=128, W_down_discriminator_size=128,
        neighbor_sampling_size=120, dataset=dataset, lr_gen=1e-4, lr_dis=1e-4,
        n_critic=5, clip_value=.03, n_epochs=epochs)

def load(instrument=False):
    env = dict(np=np, sp=sp, torch=torch, nn=nn, F=F, Variable=Variable,
               random=random, Counter=Counter, __name__='hgen_author_isolated')
    unames = {'make_noise','delete_from_tail','pad_along_axis','score_matrix_from_random_walks',
              'heterogeneous_graph_assemble','sample_from_dict','array_subtraction','symmetric'}
    nodes = [n for n in ast.parse((UP/'utils.py').read_text()).body
             if isinstance(n, ast.FunctionDef) and n.name in unames]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(UP/'utils.py'), 'exec'), env)
    env['utils'] = SimpleNamespace(**{k:env[k] for k in unames})
    hnames = {'Generator','Discriminator','train','frequent_meta_path_pattern','meta_path_frequency'}
    nodes = [n for n in ast.parse((UP/'hgen.py').read_text()).body
             if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in hnames]
    if instrument:
        train = next(n for n in nodes if n.name == 'train')
        batchloop = train.body[-1].body[0]
        assert isinstance(batchloop, ast.For)
        assert isinstance(batchloop.body[-1], ast.AugAssign)
        batchloop.body.append(ast.parse('record_update(batches_done, loss_D)').body[0])
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, str(UP/'hgen.py'), 'exec'), env)
    return env

def tail_total(rows, eos):
    """Original tail deletion on its domain; empty extension for all-EOS."""
    result=[]
    for row in rows:
        end=len(row)
        while end and row[end-1] == eos: end-=1
        result.append(row[:end].copy())
    return result

def scores_patterns(env, types, nodes, n):
    tt=tail_total(types,4); vv=tail_total(nodes,n)
    valid_t=[[x for x in tt if len(x)==k and 4 not in x] for k in [2,3,4]]
    valid_v=[[x for x in vv if len(x)==k and n not in x] for k in [2,3,4]]
    score=sp.csr_matrix((n,n))
    for rows in valid_v:
        if rows: score=score+env['score_matrix_from_random_walks'](rows,n)
    patterns=env['meta_path_frequency'](*valid_t)
    return score.tocsr(), patterns, dict(empty=int(sum(len(x)==0 for x in vv)),
        usable_lengths=[len(x) for x in valid_v], total=len(nodes))

class LazyBank(torch.utils.data.Dataset):
    def __init__(self, nodes, types, n): self.nodes,self.types,self.n=nodes,types,n
    def __len__(self): return len(self.nodes)
    def __getitem__(self,i):
        return (F.one_hot(torch.as_tensor(self.types[i]).long(),5).float(),
                F.one_hot(torch.as_tensor(self.nodes[i]).long(),self.n+1).float())

def setup(env, reference, embedding, device, epochs=5):
    typ=np.asarray(reference['node_types']);n=len(typ)
    groups={t:np.flatnonzero(typ==t).tolist() for t in range(4)}
    emb=[torch.tensor(embedding[groups[t]],dtype=torch.float32,device=device) for t in range(4)]
    a=args('IMDB_movie',epochs)
    g=env['Generator'](groups,n,emb,torch.device(device),a).to(device)
    d=env['Discriminator'](n,torch.device(device),a).to(device)
    go=torch.optim.RMSprop(g.parameters(),lr=a.lr_gen)
    do=torch.optim.RMSprop(d.parameters(),lr=a.lr_dis)
    env.update(args=a,device=torch.device(device),generator=g,discriminator=d,optimizer_G=go,optimizer_D=do)
    return g,d,go,do,groups

def graph_payload(reference, adjacency):
    """Retain forbidden type pairs; never silently project to reference schema."""
    g=copy.deepcopy(reference)
    for f in g['families']: f['edges']=[]
    bypair={tuple(sorted((f['src_type'],f['dst_type']))):f for f in g['families']}
    types=np.asarray(g['node_types']);loops=[];illegal=0
    for u,v in zip(*np.triu(np.asarray(adjacency)>0).nonzero()):
        u,v=int(u),int(v)
        if u==v: loops.append(u);continue
        pair=tuple(sorted((int(types[u]),int(types[v]))))
        if pair not in bypair:
            a,b=pair
            f=dict(name=f'unsupported_{a}_{b}',src_type=a,dst_type=b,
                   src_nodes=np.flatnonzero(types==a).tolist(),dst_nodes=np.flatnonzero(types==b).tolist(),edges=[])
            bypair[pair]=f;g['families'].append(f)
        f=bypair[pair]
        if f['name'].startswith('unsupported_'):illegal+=1
        f['edges'].append([u,v])
    return g,dict(self_loops=loops,unsupported_edges=illegal,schema_valid=illegal==0,
                  simple=len(loops)==0)
