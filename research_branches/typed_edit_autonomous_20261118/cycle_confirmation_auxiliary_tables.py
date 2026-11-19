"""Read-only export of accepted coverage-confirmation workloads and diversity."""
import json
from pathlib import Path
from confirmation_tables import ROOT,P,read,sha,write_csv
BASE=ROOT/'results/typed_edit_autonomous_20261118/cycle_repair_confirmation/auxiliary'
def main():
    acceptance=read(BASE/'acceptance.json');assert acceptance['passed'] and acceptance['evaluated_states']==28
    for path,h in read(BASE/'manifest.json')['inputs'].items():assert sha(Path(path))==h,path
    summary=read(BASE/'summary.json');rows=read(BASE/'per_output.json');assert len(rows)==28 and len(summary['groups'])==14
    flat=[{k:v for k,v in row.items() if k!='behavior'}|row['behavior'] for row in rows]
    write_csv(P/'tables/cycle_confirmation_workload_per_output.csv',flat)
    write_csv(P/'tables/cycle_confirmation_workload_summary.csv',summary['aggregate'])
    (P/'tables/cycle_confirmation_diversity.json').write_text(json.dumps(summary['groups'],indent=2)+'\n')
    labels={('F','base'):r'$L_{\mathrm{base}}$',('F','coverage'):r'$L_{\mathrm{cov}}$',('Q','fixed'):'Q',('N','random'):'N',('NS','random'):'NS',('zero','start'):'Zero'}
    lines=[r'\begin{table}[ht]',r'\centering\small',r'\caption{Coverage-confirmation workload probe. Lower discrepancy is better; this is a fixed two-hop sampling workload, not downstream predictive accuracy. Learned/Q SDs use bank means, other SDs use starts. This probe is not a model-selection gate.}',r'\label{tab:cycle-confirmation-workload}',r'\begin{tabular}{llrr}',r'\toprule',r'Prefix & Selector & Workload error & SD \\',r'\midrule']
    for r in summary['aggregate']:
        lines.append(f"{r['producer']} & {labels[r['arm'],r['variant']]} & {r['mean']:.6f} & {r['sample_sd']:.6f}"+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}'];(P/'tables/cycle_confirmation_workload.tex').write_text('\n'.join(lines)+'\n')
    receipt=dict(source_files={str(p):sha(p) for p in [BASE/'acceptance.json',BASE/'manifest.json',BASE/'summary.json',BASE/'per_output.json']},aggregate=summary['aggregate'],states=28,groups=14,new_experiments=0)
    (P/'cycle_confirmation_auxiliary_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print('Coverage workloads exported: 28 states, 14 groups')
if __name__=='__main__':main()
