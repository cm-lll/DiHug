"""Export every accepted repair-confirmation arm without replacing the old panel."""
import csv,json,statistics
from pathlib import Path
from confirmation_tables import ROOT,P,read,sha,write_csv
BASE=ROOT/'results/typed_edit_autonomous_20261118/cycle_repair_confirmation'

def main():
    source=BASE/'deploy';receipt=read(source/'acceptance.json');assert receipt['passed']
    for path,h in read(source/'manifest.json')['files'].items():assert sha(Path(path))==h,path
    for path,h in receipt['outputs'].items():assert sha(Path(path))==h,path
    summary=read(source/'summary.json');rows=summary['rows'];zero=read(source/'zero.json')
    assert len(rows)==72 and receipt['replayed_opportunities']==49152
    final=[r for r in rows if r['steps']==2048];assert len(final)==24
    aggregates=[]
    policies=[('C','F','base'),('C','F','coverage'),('C','Q','fixed'),('C','N','random'),('C','NS','random'),('D','N','random')]
    labels={('F','base'):r'$L_{\mathrm{base}}$',('F','coverage'):r'$L_{\mathrm{cov}}$',('Q','fixed'):'Q',('N','random'):'N',('NS','random'):'NS'}
    for producer,arm,variant in policies:
        rr=[r for r in final if (r['producer'],r['arm'],r['variant'])==(producer,arm,variant)]
        values=[statistics.mean(r['primary_error'] for r in rr if r['repeat']==i) for i in [1,2,3]] if arm in ['F','Q'] else [r['primary_error'] for r in rr]
        assert len(rr)==(6 if arm in ['F','Q'] else 2)
        aggregates.append(dict(dataset='imdb',producer=producer,arm=arm,variant=variant,mean=statistics.mean(values),sample_sd=statistics.stdev(values),sd_unit='three paired bank means' if arm in ['F','Q'] else 'two starts',outputs=len(rr),connected=sum(r['connected'] for r in rr),triangle_error_mean=statistics.mean(r['triangle_vector_relative_l1'] for r in rr),cycle_error_mean=statistics.mean(r['cycle_vector_relative_l1'] for r in rr),prefix_steps_mean=statistics.mean(r['first_stage_steps'] for r in rr),total_steps_mean=statistics.mean(r['total_attempts'] for r in rr)))
    for name,data in [('per_output',rows),('summary',aggregates),('paired_contrasts',summary['decisions']),('fit_means',summary['fit_means'])]:write_csv(P/f'tables/cycle_confirmation_{name}.csv',data)
    lines=[r'\begin{table}[ht]',r'\centering\small',r'\caption{Fresh-start IMDB coverage confirmation, shared C starts, 2,048 suffix opportunities. Base and coverage scorers share the corresponding original normalization with Q. Learned/Q SDs use three paired bank means over two starts; zero/N/NS SDs use two starts. Only augmentation banks and starts are fresh; base banks and the first-stage MLP are reused.}',r'\label{tab:cycle-confirmation-shared}',r'\begin{tabular}{lrrl}',r'\toprule',r'Selector & Error & SD & SD unit \\',r'\midrule']
    zz=[zero[f'imdb_{sid}_C']['primary_error'] for sid in [14001,14002]]
    lines.append(f'Zero & {statistics.mean(zz):.6f} & {statistics.stdev(zz):.6f} & Starts'+r' \\')
    for r in aggregates:
        if r['producer']!='C':continue
        lines.append(f"{labels[r['arm'],r['variant']]} & {r['mean']:.6f} & {r['sample_sd']:.6f} & "+('Bank means' if r['arm'] in ['F','Q'] else 'Starts')+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}'];(P/'tables/cycle_confirmation_shared.tex').write_text('\n'.join(lines)+'\n')
    lines=[r'\begin{table}[ht]',r'\centering\small',r'\caption{Complete IMDB policies in the coverage-confirmation panel. All suffixes use 2,048 opportunities. C/D comparison concerns complete strategies, distinct from the shared-C intervention.}',r'\label{tab:cycle-confirmation-full}',r'\begin{tabular}{lrrrr}',r'\toprule',r'Policy & Error & Prefix steps & Total steps & Connected \\',r'\midrule']
    for r in aggregates:
        lines.append(f"{r['producer']}+{labels[r['arm'],r['variant']]} & {r['mean']:.6f} & {r['prefix_steps_mean']:.0f} & {r['total_steps_mean']:.0f} & {r['connected']}/{r['outputs']}"+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}'];(P/'tables/cycle_confirmation_full.tex').write_text('\n'.join(lines)+'\n')
    lines=[r'\begin{table}[ht]',r'\centering\small',r'\caption{Every paired coverage-confirmation endpoint, lower error is better. No seed or bank is omitted.}',r'\label{tab:cycle-confirmation-pairs}',r'\begin{tabular}{rrrrrrrr}',r'\toprule',r'Start & Bank & Zero & N & NS & Q & Base & Coverage \\',r'\midrule']
    for r in summary['decisions']:
        lines.append(f"{r['seed']} & {r['repeat']} & "+' & '.join(f"{r[k]:.6f}" for k in ['zero','N','NS','Q','F_base','F_coverage'])+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}'];(P/'tables/cycle_confirmation_pairs.tex').write_text('\n'.join(lines)+'\n')
    report=dict(source=str(source),source_files={str(p):sha(p) for p in [source/'manifest.json',source/'acceptance.json',source/'summary.json',source/'zero.json']},core_gate_passed=summary['passed'],decisions=summary['decisions'],aggregates=aggregates,readouts=72,chains=24,new_experiments=0)
    (P/'cycle_confirmation_table_receipt.json').write_text(json.dumps(report,indent=2)+'\n');print('Coverage confirmation exported:',summary['passed'],sum(r['better_than_Q'] for r in summary['decisions']))

if __name__=='__main__':main()
