"""Export workload probes with the same bank-versus-start repetition units."""
import argparse,json
from pathlib import Path
from confirmation_tables import ROOT,P,read,sha,write_csv
BASE=ROOT/'results/typed_edit_crossdomain_augmentation_20261118'
def main(ds):
    source=BASE/f'confirmation/auxiliary/{ds}';a=read(source/'acceptance.json');assert a['passed']
    for p,h in read(source/'manifest.json')['inputs'].items():assert sha(Path(p))==h,p
    rows=read(source/'per_output.json');s=read(source/'summary.json');flat=[{k:v for k,v in r.items() if k!='behavior'}|r['behavior'] for r in rows]
    write_csv(P/f'tables/crossdomain_confirmation_{ds}_workload_per_output.csv',flat);write_csv(P/f'tables/crossdomain_confirmation_{ds}_workload_summary.csv',s['aggregate'])
    labels={'T':r'$L_{\mathrm{target}}$','U':r'$L_{\mathrm{ordinary}}$','B':r'$L_{\mathrm{base}}$','Q':'Q','N':'N','NS':'NS','zero':'Zero'}
    lines=[r'\begin{table}[ht]',r'\centering\small',r'\caption{PubMed fresh-start augmentation confirmation: fixed two-hop workload discrepancy. Learned/Q SDs use three paired-bank means; other SDs use starts. All endpoints and four prefix states are included. This probe does not enter model selection.}',r'\label{tab:crossdomain-'+ds+'-workload}',r'\begin{tabular}{llrr}',r'\toprule',r'Prefix & Selector & Error & SD \\',r'\midrule']
    for r in s['aggregate']:
        v='--' if r['mean'] is None else f"{r['mean']:.6f}";sd='--' if r['sample_sd'] is None else f"{r['sample_sd']:.6f}"
        lines.append(f"{r['producer']} & {labels[r['arm']]} & {v} & {sd}"+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}'];(P/f'tables/crossdomain_confirmation_{ds}_workload.tex').write_text('\n'.join(lines)+'\n')
    receipt=dict(inputs={str(p):sha(p) for p in [source/'manifest.json',source/'acceptance.json',source/'summary.json',source/'per_output.json']},aggregate=s['aggregate'],states=len(rows),groups=len(s['groups']),new_fits=0,new_graphs=0)
    (P/f'crossdomain_confirmation_{ds}_auxiliary_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(ds,'workload exported',len(rows))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset',choices=['pubmed','acm']);main(p.parse_args().dataset)
