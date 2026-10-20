"""Evaluation-only repair. Preserve original failed controller/source/all120 graphs.

Original exact dict equality compared independently ordered positive floating sums.
Validate TV against its exact integer numerator using the arithmetic forward-error
bound, not a fitted tolerance. Every other result field stays exact. No model calls.
"""
import copy
from fractions import Fraction
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'run01'
OUT=ROOT/'audit_repair_v1'
WORK=Path(os.environ.get('ROLE_REVIEW_SOURCE_ROOT','/tmp/dihug-role-unseen-target-validation-v1'))
sys.path.insert(0,str(WORK/'experiments/role_unseen_target_validation_v1'))
import analyse
from execute import load,save,digest


def valid_TV(value,actual,target,count):
    keys=actual.keys()|target.keys()
    exact=Fraction(sum(abs(actual.get(k,0)-target.get(k,0)) for k in keys),2*count)
    # k divisions + at most k positive additions, then multiplication by1/2.
    # Conservative gamma_(2k+2), standard binary64 round-to-nearest error bound.
    eps=sys.float_info.epsilon/2
    n=2*len(keys)+2
    gamma=n*eps/(1-n*eps)
    bound=Fraction.from_float(gamma)*abs(exact)
    error=abs(Fraction.from_float(float(value))-exact)
    if error>bound:raise AssertionError('TV disagrees with exact integer frequencies beyond arithmetic roundoff')
    return float(error),float(bound)


def numeric_stat(values):
    # Boolean indicators have their ordinary0/1 mean/median/linearP90.
    # NumPy1.26 cannot subtract bools in quantile interpolation.
    result=analyse_stat([float(v) if v is not None else None for v in values])
    result['values']=values
    return result


analyse_stat=analyse.stat


def main():
    started=time.monotonic();cpu=time.process_time()
    if OUT.exists():
        previous=load(OUT/'receipt.json')
        assert previous['status']=='failed'
        assert not (OUT/'attempt001_failed_receipt.json').exists()
        save(OUT/'attempt001_failed_receipt.json',previous)
        assert load(OUT/'binding.json')['activation_sha256']==digest(RUN/'activation.json')
    else:OUT.mkdir()
    deadline=load(RUN/'activation.json')['deadline_epoch']
    binding=dict(activation_sha256=digest(RUN/'activation.json'),
         original_controller_sha256=digest(RUN/'controller.json'),original_controller_status=load(RUN/'controller.json')['status'],
         original_receipt_sha256=digest(RUN/'worker_receipt.json'),same_deadline_epoch=deadline,
         no_new_activation=True,no_model_calls=True,original_STOP_retained=True,
         change='exact cross-process dict equality replaced ONLY for jointTV by exact-integer forward-error check')
    if not (OUT/'binding.json').exists():save(OUT/'binding.json',binding)
    else:assert load(OUT/'binding.json')==binding
    def check():
        if time.time()>=deadline or (ROOT/'AUDIT_STOP').exists():raise TimeoutError('original45min deadline or AUDIT_STOP')
    original=analyse.independently_check;old_save=analyse.save;rounding=[]
    def independent(row,out,target):
        tmp=copy.deepcopy(row)
        expected=analyse.target_errors(row['metrics'],target)
        for typ,name in enumerate(analyse.NAMES):
            actual=row['metrics']['joint_relation_degree'][name]
            wanted={','.join(map(str,v)):int(n) for v,n in target['roles'][typ]}
            error,bound=valid_TV(row['target_errors'][name]['joint_TV'],actual,wanted,target['type_counts'][typ])
            rounding.append(dict(graph_file=row['graph_file'],type=name,absolute_error_to_exact_fraction=error,bound=bound))
            tmp['target_errors'][name]['joint_TV']=expected[name]['joint_TV']
        # Full original independent schema/degree/path/closure/W1 audit, no bypass.
        return original(tmp,out,target)
    def redirected(path,value):
        assert path.parent==RUN
        old_save(OUT/path.name,value)
    status='failed';reason=None
    try:
        assert load(RUN/'controller.json')['status']=='failed'
        check();analyse.independently_check=independent;analyse.save=redirected;analyse.stat=numeric_stat
        analyse.audit(RUN,check);check()
        save(OUT/'roundoff.json',rounding);status='completed'
    except BaseException as exc:
        reason=repr(exc);traceback.print_exc()
    finally:
        analyse.independently_check=original;analyse.save=old_save;analyse.stat=analyse_stat
        save(OUT/'receipt.json',dict(status=status,reason=reason,wall_seconds=time.monotonic()-started,
             cpu_seconds=time.process_time()-cpu,elapsed_from_original_activation=time.time()-load(RUN/'activation.json')['activation_epoch'],
             max_observed_TV_roundoff=max((r['absolute_error_to_exact_fraction'] for r in rounding),default=None),
             original_failure_preserved=True,graphs_regenerated=0,model_calls=0,training_updates=0))
    return 0 if status=='completed' else 1

if __name__=='__main__':sys.exit(main())
