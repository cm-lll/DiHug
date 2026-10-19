"""SYNTHETIC standard-format fixture writer; NOT an old real-output converter."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from joint_decode.contracts import Schema
from joint_decode.frozen_input import array_sha,query_plan,FORMAT


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def write_json(p,value):p.write_text(json.dumps(value,indent=2)+'\n')


def fixture(root,arm='A',reorder=False,original=True):
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    s=Schema(np.array([0,0,1,1,2,2]),np.array(list(itertools.combinations_with_replacement(range(3),2))))
    bits=np.zeros(s.s,np.uint8);bits[s.encode_edges(np.array([[0,1,2,3,0,1],[4,5,4,5,2,3]]))]=1
    profile=s.profile(bits);counts=np.bincount(s.families,minlength=s.r);edge=np.bincount(s.families,weights=bits,minlength=s.r)
    prior=np.divide(edge,counts,out=np.zeros(s.r),where=counts>0);support=np.where(prior[s.families]==0,0,np.where(prior[s.families]==1,1,-1)).astype(np.int8)
    p0=np.linspace(.19,.81,s.s);p0[support!=-1]=support[support!=-1]
    seed=2026101907;plan=query_plan(s,edge>0,seed);alpha=np.linspace(1,0,1001,dtype=np.float64)
    # Deliberately SYNTHETIC schedule, bound by receipt; never called the actual model schedule.
    state=np.zeros(s.s,np.uint8);state[support==1]=1
    arrays=dict(types=s.types,pairs=s.pairs,slots=s.slots,families=s.families,state=state,p0=p0,q=p0.copy(),
                support=support,prior=prior,alpha=alpha,**{k:v for k,v in plan.items() if k!='query_sha256'})
    if reorder:
        order=np.arange(s.s)[::-1]
        for k in ('families','state','p0','q','support','query_block'):arrays[k]=arrays[k][order]
        arrays['slots']=arrays['slots'][:,order]
    np.savez_compressed(root/'payload.npz',**arrays)
    receipt=dict(arm=arm,seed=seed,t=1,unguided=True,query_sha256=plan['query_sha256'],source_kind='synthetic_fixture',
        weight_authentication='historical_receipt_only',weight_sha256='1'*64,source_manifest_sha256='2'*64,
        config_sha256='3'*64,reference_sha256='4'*64,alpha_sha256=array_sha(alpha),
        note='ALL DATA SYNTHETIC; dummy hashes are NOT authenticated model/source evidence')
    write_json(root/'receipt.json',receipt);write_json(root/'target.json',dict(profile=profile,reference_sha256='4'*64))
    manifest=dict(format=FORMAT,arm=arm,t=1,unguided=True,seed=seed,T=1000,query_rounds=20,
        p0_semantics='clean_edge_present_probability',score_semantics='recorded_probability',
        q_semantics='next_state_edge_present_probability',endpoint_convention='unordered_u_lt_v',
        relation_ids='zero_based_family_pairs_order',query_sha256=plan['query_sha256'],semantic_types=dict(director=0,actor=1,movie=2),
        files={k:dict(path=n,sha256=sha(root/n)) for k,n in dict(payload='payload.npz',target='target.json',source_receipt='receipt.json').items()})
    write_json(root/'manifest.json',manifest);pins=dict(manifest_sha256=sha(root/'manifest.json'))
    if original:
        np.savez_compressed(root/'original.npz',types=s.types,edges=s.slots[:,bits.astype(bool)],families=s.families[bits.astype(bool)])
        orig_receipt=dict(arm=arm,seed=seed,t=0,unguided=True,graph_stage='raw_final_t0',
             input_manifest_sha256=pins['manifest_sha256'],source_receipt_sha256=manifest['files']['source_receipt']['sha256'],
             graph_sha256=sha(root/'original.npz'),weight_sha256=receipt['weight_sha256'])
        write_json(root/'original_receipt.json',orig_receipt)
        pins.update(original_sha256=sha(root/'original.npz'),original_receipt_sha256=sha(root/'original_receipt.json'))
    write_json(root/'fixture_pins.json',pins)
    return pins


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);a=p.parse_args()
    print(json.dumps(fixture(a.output)))
