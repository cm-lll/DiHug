"""Complete-policy comparison with every external version and conditioning caveat."""
import csv,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];P=ROOT/'results/original_paper_revision_20261118_potential'
R=ROOT/'results/typed_edit_autonomous_20261118'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    current=read(P/'confirmation_table_receipt.json');external=read(P/'harmonized_external_receipt.json')
    for receipts,key in [(current,'source_files'),(external,'inputs')]:
        for p,h in receipts[key].items():assert sha(Path(p))==h,p
    assert read(R/'confirmation_v1/deploy/acceptance.json')['passed']
    assert read(R/'external_recount_finalized/acceptance.json')['passed']
    external_rows=read(R/'external_recount_finalized/summary.json')['groups'];rows=[]
    for r in current['aggregates']:
        rows.append(dict(dataset=r['dataset'],method='DiHug '+r['producer']+'+'+r['selector'],
            mean=r['mean'],sample_sd=r['sample_sd'],sd_unit=r['sd_unit'],
            valid=r['outputs'],requests=r['outputs'],whole_connected=r['connected'],
            source_category='revised complete policy',alias=False,
            conditioning='anonymous joint relation degrees; reference-derived training normalization for F/Q',
            budget='prefix up to arrival + fixed 2048 suffix opportunities',matched_external_compute=False))
    for r in external_rows:
        rows.append(dict(dataset=r['dataset'],method=r['variant'],mean=r['conditional_fit_mean'],
            sample_sd=r['conditional_fit_sample_sd'],sd_unit='fit means over eligible assembly outputs' if r['fit_count']>1 else 'single fit; no training SD',
            valid=r['eligible'],requests=r['logical_requests'],whole_connected=r['whole_connected'],
            source_category='historical reconstruction' if r['variant'].startswith('reconstruction') else 'historical official core or adaptation',
            alias=r['reused_alias'],conditioning='native count/embedding and assembly conditions; no matched anonymous degree target',
            budget='retained historical fit and assembly budgets',matched_external_compute=False))
    assert len(rows)==30
    with (P/'tables/complete_external_comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    names={'DiHug C+F':r'DiHug C+$L_{\mathrm{full}}$','DiHug C+Q':'DiHug C+Q','DiHug D+N':'DiHug D+N',
           'reconstruction_r2_walk_reused':'HGEN walk reconstruction',
           'reconstruction_r3_motif':'HGEN motif reconstruction',
           'official_core_strict':'HGEN official core, strict',
           'official_core_compatible':'HGEN official core, compatible',
           'adapter_G_strict':'HGEN gradient adaptation, strict',
           'adapter_G_compatible':'HGEN gradient adaptation, compatible',
           'adapter_S_strict':'HGEN schema adaptation, strict',
           'adapter_S_compatible':'HGEN schema adaptation, compatible'}
    selected=[]
    for ds in ['imdb','pubmed']:
        lines=[r'\begin{table}[ht]',r'\centering\small',r'\setlength{\tabcolsep}{3pt}',
            r'\caption{'+ds.upper()+r' complete-policy comparison under the same structural metric. Lower is better. Native conditioning and compute budgets differ; this is not a matched-information or matched-compute experiment. External means condition on legal completion, with every request retained in the denominator. Learned DiHug SD uses three corruption-bank fit means over two starts; Q uses the same bank normalizations without fitting; D+N SD uses two starts. External SD uses historical fit means, and is absent for one-fit adaptations. Aliases and the additional C+N/NS controls remain in the accompanying full data.}',
            r'\label{tab:complete-external-'+ds+'}',r'\begin{tabular}{p{.51\linewidth}rrr}',r'\toprule',
            r'Complete policy & Valid/req. & Conn. & Error (mean $\pm$ SD) \\',r'\midrule']
        for method,label in names.items():
            rr=[r for r in rows if r['dataset']==ds and r['method']==method];assert len(rr)==1
            r=rr[0];assert not r['alias'];selected.append(r)
            cell='--' if r['mean'] is None else f"${r['mean']:.4f}$" if r['sample_sd'] is None else f"${r['mean']:.4f}\\pm{r['sample_sd']:.4f}$"
            lines.append(f"{label} & {r['valid']}/{r['requests']} & {r['whole_connected']} & {cell}"+r' \\')
        lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
        (P/f'tables/complete_external_{ds}.tex').write_text('\n'.join(lines)+'\n')
    assert len(selected)==22
    inputs=[Path(__file__),P/'confirmation_table_receipt.json',P/'harmonized_external_receipt.json',R/'external_recount_finalized/summary.json']
    outputs=[P/'tables/complete_external_comparison.csv',P/'tables/complete_external_imdb.tex',P/'tables/complete_external_pubmed.tex']
    (P/'complete_external_receipt.json').write_text(json.dumps(dict(passed=True,all_rows=30,displayed_rows=22,
        new_experiments=0,matched_external_compute=False,matched_external_conditioning=False,
        inputs={str(p):sha(p) for p in inputs},outputs={str(p):sha(p) for p in outputs}),indent=2)+'\n')
    print('Complete external comparison:',30,'rows; no version mixing')
if __name__=='__main__':main()
