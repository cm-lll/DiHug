"""Accepted confirmation workload and two-start diversity; no new outputs."""
import csv,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];P=ROOT/'results/original_paper_revision_20261118_potential'
S=ROOT/'results/typed_edit_autonomous_20261118/confirmation_v1/auxiliary'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def csvout(name,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (P/'tables'/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def main():
    receipt=read(S/'acceptance.json');assert receipt['passed'] and receipt['evaluated_states']==44
    for p,h in read(S/'manifest.json')['inputs'].items():assert sha(Path(p))==h,p
    summary=read(S/'summary.json');per=read(S/'per_output.json');assert len(per)==44
    flat=[]
    for row in per:
        r={k:v for k,v in row.items() if k!='behavior'};r.update(row['behavior']);flat.append(r)
    csvout('confirmation_workload_per_output.csv',flat)
    csvout('confirmation_workload_summary.csv',summary['aggregate'])
    (P/'tables/confirmation_diversity.json').write_text(json.dumps(summary['groups'],indent=2)+'\n')
    labels={'F':r'$L_{\mathrm{full}}$','Q':'Q','N':'N','NS':'NS','zero':'Zero'}
    lines=[r'\begin{table}[ht]',r'\centering\small',r'\caption{Unchanged two-hop workload probe on all 36 frozen-confirmation endpoints and eight zero-step states. Lower is better. F/Q uncertainty uses three bank means over two starts; other SDs use two starts. This probe is not used for model selection.}',r'\label{tab:confirmation-workload}',r'\begin{tabular}{llrr}',r'\toprule',r'Reference & Policy & Mean error & SD \\',r'\midrule']
    for r in summary['aggregate']:
        lines.append(f"{r['dataset']} & {r['producer']}+{labels[r['arm']]} & {r['mean']:.6f} & {r['sample_sd']:.6f}"+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    (P/'tables/confirmation_workload.tex').write_text('\n'.join(lines)+'\n')
    inputs=[Path(__file__),S/'manifest.json',S/'acceptance.json',S/'summary.json',S/'per_output.json']
    (P/'confirmation_auxiliary_receipt.json').write_text(json.dumps(dict(passed=True,new_experiments=0,
        states=44,groups=len(summary['groups']),aggregate=summary['aggregate'],
        inputs={str(p):sha(p) for p in inputs}),indent=2)+'\n')
    print('Confirmation auxiliary tables exported:',44,'states')
if __name__=='__main__':main()
