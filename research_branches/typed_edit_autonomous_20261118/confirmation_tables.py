"""Accepted frozen confirmation: separate shared-start and full-policy tables."""
import csv,hashlib,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'results/typed_edit_autonomous_20261118/confirmation_v1'
P=ROOT/'results/original_paper_revision_20261118_potential'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def write_csv(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def main():
    source=BASE/'deploy';receipt=read(source/'acceptance.json');assert receipt['passed']
    for name,h in read(source/'manifest.json')['files'].items():assert sha(Path(name))==h,name
    for name,h in receipt['outputs'].items():assert sha(Path(name))==h,name
    summary=read(source/'summary.json');rows=summary['rows'];zero=read(source/'zero.json')
    assert len(rows)==108 and receipt['replayed_opportunities']==73728
    final=[r for r in rows if r['steps']==2048];assert len(final)==36
    aggregates=[]
    for ds in ['imdb','pubmed']:
        for producer,arms in [('C',['F','Q','N','NS']),('D',['N'])]:
            for arm in arms:
                selected=[r for r in final if (r['dataset'],r['producer'],r['arm'])==(ds,producer,arm)]
                repeats=sorted({r['repeat'] for r in selected})
                if arm in ['F','Q']:
                    values=[statistics.mean(r['primary_error'] for r in selected if r['repeat']==rep) for rep in repeats]
                    assert len(values)==3 and len(selected)==6;unit='bank means over two starts'
                else:values=[r['primary_error'] for r in selected];assert len(values)==2;unit='two starts'
                aggregates.append(dict(dataset=ds,producer=producer,selector=arm,mean=statistics.mean(values),sample_sd=statistics.stdev(values),sd_unit=unit,outputs=len(selected),coefficient_fits=3 if arm=='F' else 0,connected=sum(r['connected'] for r in selected),triangle_error_mean=statistics.mean(r['triangle_vector_relative_l1'] for r in selected),cycle_error_mean=statistics.mean(r['cycle_vector_relative_l1'] for r in selected),first_stage_mean=statistics.mean(r['first_stage_steps'] for r in selected),suffix_opportunities=2048,total_attempts_mean=statistics.mean(r['total_attempts'] for r in selected)))
    write_csv(P/'tables/confirmation_per_output.csv',rows)
    write_csv(P/'tables/confirmation_summary.csv',aggregates)
    write_csv(P/'tables/confirmation_paired_contrasts.csv',summary['decisions'])
    write_csv(P/'tables/confirmation_fit_means.csv',summary['fit_means'])
    labels={'F':r'$L_{\mathrm{full}}$','Q':'Q','N':'N','NS':'NS'}
    lines=[r'\begin{table}[ht]',r'\centering\small',r'\caption{Frozen confirmation, shared C starts: structural error (lower is better). Learned and Q SDs use three bank means, each averaged over two starts; zero/N/NS SDs use two starts. Q has no inverse-label fitting. The old training bank and first-stage MLP are shared across bank repeats.}',r'\label{tab:frozen-shared}',r'\begin{tabular}{llrrl}',r'\toprule',r'Reference & Selector & Mean & SD & SD unit \\',r'\midrule']
    for ds in ['imdb','pubmed']:
        values=[zero[f'{ds}_{sid}_C']['primary_error'] for sid in [12001,12002]]
        lines.append(f'{ds} & Zero & {statistics.mean(values):.6f} & {statistics.stdev(values):.6f} & Starts'+r' \\')
        for arm in ['N','NS','Q','F']:
            r=next(r for r in aggregates if (r['dataset'],r['producer'],r['selector'])==(ds,'C',arm))
            unit='Bank means' if arm in ['F','Q'] else 'Starts'
            lines.append(f"{ds} & {labels[arm]} & {r['mean']:.6f} & {r['sample_sd']:.6f} & {unit}"+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    (P/'tables/confirmation_shared.tex').write_text('\n'.join(lines)+'\n')
    lines=[r'\begin{table}[ht]',r'\centering\small',r'\caption{Complete frozen policies. All outputs include first-stage initialization and editing. C uses the original MLP and calibrated degree term; D uses no learned score and a fixed degree coefficient. Comparisons between C and D therefore concern complete strategies. The suffix budget is 2,048 for all rows.}',r'\label{tab:frozen-full}',r'\begin{tabular}{llrrrr}',r'\toprule',r'Reference & Policy & Error & Prefix steps & Total steps & Connected \\',r'\midrule']
    for r in aggregates:
        policy=r['producer']+'+'+labels[r['selector']]
        lines.append(f"{r['dataset']} & {policy} & {r['mean']:.6f} & {r['first_stage_mean']:.0f} & {r['total_attempts_mean']:.0f} & {r['connected']}/{r['outputs']}"+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    (P/'tables/confirmation_full.tex').write_text('\n'.join(lines)+'\n')
    report=dict(source=str(source),source_files={str(p):sha(p) for p in [source/'manifest.json',source/'acceptance.json',source/'summary.json',source/'zero.json']},core_gate_passed=summary['passed'],decisions=summary['decisions'],aggregates=aggregates,per_output_rows=len(rows),unique_suffix_chains=len(final),new_experiments=0)
    (P/'confirmation_table_receipt.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Frozen confirmation tables:',len(rows),'readouts;',len(final),'chains; core gate',summary['passed'])

if __name__=='__main__':main()
