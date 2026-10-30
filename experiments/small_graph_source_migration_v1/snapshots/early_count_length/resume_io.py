"""Explicit compatible-parent restoration; no discarded source checks."""
import pathlib,copy
from model_io import ROOT,tensor_hash
from runtime_contract import read,sha,PROTOCOL,cell,dictionary_expected
import numpy as np
import torch
from prepare import load_case
from model_variants import StructuralDenoiser
from early_cn import install
from experiment_contract import read_dictionary
from ema import ParameterEMA
from integration import restore_payload
from probe import rng_get,rng_set,equal,input_hash
from normal2 import pair_inputs

def validate_parent(payload,expected,job,cfg):
    contract=read(ROOT/'parent_contract.json')
    if payload['binding']!=expected:raise ValueError('parent sidecar/payload mismatch')
    b=payload['binding']
    if (b['dataset'],b['train_seed'],b['arm'])!=(job['dataset'],job['train_seed'],job['arm']):raise ValueError('parent identity')
    if payload['updates']!=500 or payload['ema']['updates']!=500 or b['updates']!=500:raise ValueError('resume must start at500')
    if b['source_sha256']!=contract['source_sha256'] or b['protocol_sha256']!=contract['protocol_sha256']:raise ValueError('parent source/protocol')
    if b['formula_sha256']!=contract['formula_sha256'] or b['loss']!=contract['loss'] or b['approval_sha256']!=contract['approval_sha256']:raise ValueError('parent formula/authorization')
    if b['config']!=cfg or payload['ema']['decay']!=.99:raise ValueError('parent scientific configuration')
    weight=0. if job['arm']=='CE' else .001
    if b['lambda_count']!=weight:raise ValueError('parent loss-arm identity')
    if set(payload['rng'])!={'numpy','global_numpy','python','torch','device'}:raise ValueError('incomplete RNG')
    return b

def parent_files(job):
    row=read(ROOT/'parent_checkpoints.json')[job['id']]
    if sha(row['path'])!=row['sha256'] or sha(row['binding_path'])!=row['binding_sha256']:raise ValueError('parent file hash')
    return row

def resume_parent(job,device='cuda'):
    row=parent_files(job);payload=torch.load(row['path'],map_location='cpu');expected=read(row['binding_path'])
    case,cfg=load_case(job['dataset']);cfg=dict(cfg,seed=job['train_seed'])
    validate_parent(payload,expected,job,cfg)
    basis,receipt=read_dictionary(ROOT/'dictionaries',job['dataset'],dictionary_expected(job['dataset']))
    # Construction creates tensor storage, then every learned/buffer value is
    # replaced by raw500. All construction RNG effects are also replaced.
    model=install(StructuralDenoiser(case,cfg,'E',basis)).to(device)
    model.count_aux_weight=expected['lambda_count']
    opt=torch.optim.AdamW(model.parameters(),**{k:v for k,v in cfg['optimizer'].items() if k!='type'})
    ema=ParameterEMA(model);rng=np.random.default_rng(0)
    restored=restore_payload(payload,model,opt,ema,expected,lambda state:rng_set(state,rng,device))
    exact={k:equal(v,payload[k]) for k,v in [('model',model.state_dict()),('optimizer',opt.state_dict()),('ema',ema.state_dict()),('rng',rng_get(rng,device))]}
    if restored!=500 or not all(exact.values()):raise ValueError('inexact checkpoint load')
    if {id(p) for g in opt.param_groups for p in g['params']}!={id(p) for p in model.parameters() if p.requires_grad}:raise ValueError('optimizer coverage')
    proof=dict(parent=row,loaded_updates=restored,load_exact=exact,raw_loaded_not_ema=True,
               construction_random_states_replaced=True,optimizer_reset_after_restore=False,
               learning_rate=opt.param_groups[0]['lr'],ema_updates=ema.updates,
               parent_binding=expected,parameters=sum(p.numel() for p in model.parameters()))
    return model,opt,ema,case,cfg,rng,proof
