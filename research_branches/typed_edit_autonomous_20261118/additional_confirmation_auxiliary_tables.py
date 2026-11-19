"""Export accepted additional-reference probe values without selecting models."""
import argparse,json
from pathlib import Path
from confirmation_tables import ROOT,P,read,sha,write_csv
BASE=ROOT/'results/typed_edit_autonomous_20261118/additional_confirmation_v1'
def main(ds):
    parent=BASE/f'auxiliary/{ds}';a=read(parent/'acceptance.json');assert a['passed']
    for p,h in read(parent/'manifest.json')['inputs'].items():assert sha(Path(p))==h,p
    rows=read(parent/'per_output.json');summary=read(parent/'summary.json');assert len(rows)==a['evaluated_states']
    flat=[{k:v for k,v in r.items() if k!='behavior'}|r['behavior'] for r in rows]
    write_csv(P/f'tables/additional_confirmation_{ds}_workload_per_output.csv',flat)
    write_csv(P/f'tables/additional_confirmation_{ds}_workload_summary.csv',summary['aggregate'])
    (P/f'tables/additional_confirmation_{ds}_diversity.json').write_text(json.dumps(summary['groups'],indent=2)+'\n')
    labels={'F':r'$L_{\mathrm{full}}$','Q':'Q','N':'N','NS':'NS','zero':'Zero'}
    lines=[r'\begin{table}[ht]',r'\centering\small',r'\caption{'+ds.upper()+r' frozen-confirmation workload probe (lower discrepancy is better). F/Q SDs use three bank means; other SDs use starts. All endpoints and four zero-step prefix states are retained. This is a fixed neighbor-sampling workload, not predictive accuracy or a model-selection criterion.}',r'\label{tab:additional-confirmation-'+ds+'-workload}',r'\begin{tabular}{llrr}',r'\toprule',r'Prefix & Selector & Workload error & SD \\',r'\midrule']
    for r in summary['aggregate']:
        value='--' if r['mean'] is None else f"{r['mean']:.6f}";sd='--' if r['sample_sd'] is None else f"{r['sample_sd']:.6f}"
        lines.append(f"{r['producer']} & {labels[r['arm']]} & {value} & {sd}"+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}'];(P/f'tables/additional_confirmation_{ds}_workload.tex').write_text('\n'.join(lines)+'\n')
    receipt=dict(inputs={str(p):sha(p) for p in [parent/'manifest.json',parent/'acceptance.json',parent/'summary.json',parent/'per_output.json']},aggregate=summary['aggregate'],states=len(rows),groups=len(summary['groups']),new_experiments=0)
    (P/f'additional_confirmation_{ds}_auxiliary_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(ds,'workload export',len(rows))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset',choices=['acm','freebase']);main(p.parse_args().dataset)
