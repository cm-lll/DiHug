"""Same augmentation producer with the corrected independent replay checker."""
import argparse,types
import crossdomain_augmentation_cache as original
from crossdomain_augmentation_finalize_v2 import audit
from round01 import b

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset',choices=['pubmed','acm']);p.add_argument('--bank',type=int,choices=[2,3,4],required=True);a=p.parse_args()
    pilot=original.BASE/f'deploy/{a.dataset}/bank1'
    assert b.read(pilot/'acceptance.json')['passed'] and b.read(pilot/'summary.json')['passed']
    def bind(out,files,**metadata):
        extra=[original.SRC/'CROSSDOMAIN_AUGMENTATION_CONFIRMATION.md',pilot/'manifest.json',pilot/'acceptance.json',pilot/'summary.json']
        original.contract.bind(out,[*files,*extra],**metadata)
    proxy=types.SimpleNamespace(bind=bind,put=original.contract.put,verify=original.contract.verify)
    scope=dict(original.main.__globals__,audit_trajectory=audit,contract=proxy)
    run=types.FunctionType(original.main.__code__,scope,'confirmed_augmentation_cache',original.main.__defaults__)
    try:run(a.dataset,'cache',a.bank)
    except BaseException as exc:
        out=original.BASE/f'cache/{a.dataset}/bank{a.bank}'
        if out.exists():original.contract.put(out,'failure.json',dict(type=type(exc).__name__,message=str(exc)))
        raise
