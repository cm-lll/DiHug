"""Seal already prepared data/config/source after zero-skip gates; no learning."""
import xml.etree.ElementTree as ET
from common import *

def main():
    check();assert not (RESULT/'freeze.json').exists()
    assert read(RESULT/'correctness.json')['status']=='passed'
    assert read(RESULT/'preparation_receipt.json')['status']=='completed'
    xml=ET.parse(RESULT/'tests.xml')
    assert len(xml.findall('.//testcase'))==33
    assert not xml.findall('.//failure')+xml.findall('.//error')+xml.findall('.//skipped')
    current=settings();previous=read(RESULT/'resolved_config.json')
    if previous!=current:
        # Preserve preparation metadata; only remove inherited unused old-run budget/holdout keys.
        save(RESULT/'preparation_resolved_config.json',previous)
        assert previous['model']==current['model'] and previous['optimizer']==current['optimizer']
        assert all(previous[k]==current[k] for k in ('input_seed','graph_seed','augmentation_seed','selection_seed','T','rounds','updates'))
        save(RESULT/'resolved_config.json',current)
    inputs={n:sha(RESULT/n) for n in ('activation.json','conditions.json','panel.json','construction.json','split_binding.json','correctness.json','resolved_config.json','tests.xml')}
    save(RESULT/'freeze.json',dict(source=source_manifest(),inputs=inputs,config=current,time_epoch=time.time(),
         source_commit='recorded in formal_launch.json after commit',original_init_manifest=read(EXP/'role_specialization_v1_results/evidence/learning_run01/weight_hashes.json'),
         tests=33,zero_skips=True,method_unchanged=True))
    check_binding();print('FROZEN',sha(RESULT/'freeze.json'),'elapsed',time.time()-read(RESULT/'activation.json')['activation_epoch'])

if __name__=='__main__':main()
