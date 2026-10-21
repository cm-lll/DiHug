"""Budget-only stage boundary guard, no model/config/numerical changes.

User requires stop on expected item overrun. Wait for fully saved cold proof;
use90% of measured cold minus convolution as optimistic fresh-repeat forecast.
If even this exceeds frozen remaining420s item budget,request STOP,never extend.
"""
import time,json,hashlib
from pathlib import Path
OUT=Path('/data2/lyh/gnn_project/DiHug/results/role_newton_large_model_20261021_v1')
def main():
    start=time.time();ledger=json.loads((OUT/'activation.json').read_text());folder=OUT/'PubMed_t20'
    while time.time()<ledger['deadline_epoch']-720 and time.time()-start<300:
        if (OUT/'controller.json').exists():return
        p=folder/'paths.json';progress=folder/'progress.json'
        if p.exists():
            rows=json.loads(p.read_text());cold=next(x for x in rows if x['phase']=='cold')
            elapsed=json.loads(progress.read_text())['elapsed_seconds'] if progress.exists() else cold['full_step_seconds']
            optimistic=.9*(cold['full_step_seconds']-cold['module_timings'].get('degree_law_cache_construction_seconds',0))
            remaining=420-elapsed-3
            receipt=dict(scope='budget supervision only,unchanged frozen numerical/scientific settings',cold_complete_evidence=True,
                cold_seconds=cold['full_step_seconds'],optimistic_fresh_repeat_estimate=optimistic,remaining_frozen_item_seconds=remaining,
                optimistic_factor=.9,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),epoch=time.time())
            if optimistic>remaining:
                receipt['decision']='STOP projected item overrun; no parameter/state substitution,do not launch additional PubMed probes'
                (OUT/'forecast_stop_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
                (OUT/'STOP').write_text(json.dumps(receipt)+'\n');return
            receipt['decision']='continue within existing gate';(OUT/'forecast_continue_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');return
        time.sleep(.1)
if __name__=='__main__':main()
