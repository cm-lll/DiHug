"""Small deterministic ENGINEERING test only; never a real-data configuration."""
import copy
from pathlib import Path
from hsgd.fusion_checks import fixture as core_fixture
from .data import load

def fixture():
    case,_,_=core_fixture();_,cfg=load('IMDB');cfg=copy.deepcopy(cfg)
    cfg.update(dataset='ENGINEERING_ONLY',T=4,rounds=3,max_updates=4,
               checkpoint_every_updates=1,validation_every_updates=2,validation_times=[1,2,4],evaluation_updates=[2,4])
    cfg['model'].update(layers=2,width=16,edge_width=8,global_width=8,heads=2,ff_node=16,ff_edge=8,ff_global=8)
    cfg['data'].update(type_names=['x','y','z'])
    return case,cfg

if __name__=='__main__':
    import argparse
    from .engine import run_train,sample,preflight
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['train','resume','sample','gate']);p.add_argument('out',type=Path);p.add_argument('--run',type=Path)
    a=p.parse_args();case,cfg=fixture()
    if a.mode=='gate':preflight(case,cfg,a.out,'cpu')
    elif a.mode=='sample':
        if not sample(case,cfg,a.run,a.out,4,'cpu'):raise SystemExit(2)
    else:
        if not run_train(case,cfg,a.out,'cpu',fresh=a.mode=='train',resume=a.mode=='resume',accept_gap=True):raise SystemExit(2)
