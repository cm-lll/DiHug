"""Tiny synthetic probability FILE fixtures, never actual weights/model calls."""
import copy
import json
import numpy as np
import pytest
from hsgd.bw_preflight import common as c
from hsgd.kernel import MarginalKernel
from stepwise_role.core import Layout
from multiscale_role.short import array_sha,file_sha,validate_record,verify_original_q


def saved_fixture(folder):
    case=c.toy_case();l=Layout.canonical(case['space'].types,case['space'].pairs,case['prior'])
    p=np.repeat(case['prior'],case['space'].counts);state=np.zeros(len(p),dtype=np.uint8);state[case['clean_ids']]=1
    k=MarginalKernel(case['prior'],c.schedule(1000));t=20;seed=12345
    plan=c.plan_for(case,[seed,t,61],20);blocks=np.full(len(p),-1,dtype=int)
    for r in range(20):blocks[plan.block(r)[0]]=r
    data=dict(state=state,p0=p,q=k.reverse(p,state,l.families,t-1,t),slots=l.slots,families=l.families,
              types=l.types,prior=l.prior,alpha=k.alpha,query_block=blocks,t=np.array(t))
    row=dict(arm='A',t=t,reference_sha256='toy_reference',npz='saved.npz',metadata='meta.json')
    meta=dict(arm='A',model_arm='A',t=t,T=1000,query_rounds=20,sampling_seed=seed,
              reference_sha256='toy_reference',completed_updates=6000,final_weight=True)
    for name in ('checkpoint','source_manifest','config'):
        path=folder/(name+'.fixture');path.write_text('NOT_A_REAL_MODEL: deterministic engineering artifact only')
        row[name]=path.name;row[name+'_sha256']=file_sha(path);meta[name+'_sha256']=row[name+'_sha256']
    return case,l,k,data,row,meta


def freeze_files(folder,data,row,meta):
    np.savez(folder/row['npz'],**data)
    meta['array_sha256']={k:array_sha(a) for k,a in data.items()}
    (folder/row['metadata']).write_text(json.dumps(meta))
    row['metadata_sha256']=file_sha(folder/row['metadata']);row['npz_sha256']=file_sha(folder/row['npz'])


def test_saved_probability_bindings_and_original_q(tmp_path):
    case,l,k,data,row,meta=saved_fixture(tmp_path);freeze_files(tmp_path,data,row,meta)
    got,_=validate_record(row,tmp_path,case,lambda:None,{})
    _,q,error=verify_original_q(got,k,l,20)
    assert error==0 and np.array_equal(q,data['q'])
    got['q']=got['q'].copy();got['q'][0]+=.001
    with pytest.raises(ValueError,match='saved q'):verify_original_q(got,k,l,20)
    (tmp_path/row['npz']).write_bytes(b'changed-original-input')
    with pytest.raises(ValueError,match='NPZ hash'):validate_record(row,tmp_path,case,lambda:None,{})


@pytest.mark.parametrize('field',['slots','query_block','alpha','prior','types'])
def test_even_rehashed_wrong_mapping_or_schedule_rejected(tmp_path,field):
    case,l,k,data,row,meta=saved_fixture(tmp_path)
    if field=='slots':data[field]=data[field][:,::-1].copy()
    elif field=='alpha':data[field]=data[field].copy();data[field][500]+=.001
    elif field=='prior':data[field]=data[field].copy();data[field][0]+=.001
    elif field=='types':data[field]=data[field].copy();data[field][0]=1
    else:data[field]=data[field].copy();data[field][0]=(data[field][0]+1)%20
    freeze_files(tmp_path,data,row,meta)
    with pytest.raises(ValueError):validate_record(row,tmp_path,case,lambda:None,{})
