"""Export a complete accepted fresh-bank panel, retaining every unavailable start."""
import argparse,json,statistics
from pathlib import Path
from confirmation_tables import ROOT,P,read,sha,write_csv
BASE=ROOT/'results/typed_edit_autonomous_20261118/additional_confirmation_v1'
def main(ds):
    source=BASE/f'deploy/{ds}';accept=read(source/'acceptance.json');assert accept['passed']
    for path,h in read(source/'manifest.json')['files'].items():assert sha(Path(path))==h,path
    for path,h in accept['outputs'].items():assert sha(Path(path))==h,path
    summary=read(source/'summary.json');zero=read(source/'zero.json');rows=summary['rows'];final=[r for r in rows if r['steps']==2048]
    assert len(rows)==3*len(final) and accept['replayed_opportunities']==2048*len(final)
    pr=read(BASE/f'prefix/{ds}/summary.json')['rows'];aggregates=[];fitmeans=[]
    for producer,arms in [('C',['zero','F','Q','N','NS']),('D',['N'])]:
        arrived=[r['seed'] for r in pr if r['arm']==producer and r['arrived']]
        for arm in arms:
            selected=[r for r in final if (r['producer'],r['arm'])==(producer,arm)]
            if arm=='zero':values=[zero[f'{sid}_C']['primary_error'] for sid in arrived]
            elif arm in ['F','Q']:
                values=[]
                for repeat in [2,3,4]:
                    rr=[r for r in selected if r['repeat']==repeat]
                    if rr:
                        v=statistics.mean(r['primary_error'] for r in rr);values.append(v)
                        fitmeans.append(dict(dataset=ds,arm=arm,repeat=repeat,eligible_starts=len(rr),mean=v))
            else:values=[r['primary_error'] for r in selected]
            aggregates.append(dict(dataset=ds,producer=producer,arm=arm,mean=statistics.mean(values) if values else None,sample_sd=statistics.stdev(values) if len(values)>1 else None,sd_unit='three new-bank means over eligible starts' if arm in ['F','Q'] else 'starts',eligible=len(arrived) if arm=='zero' else len(selected),planned=6 if arm in ['F','Q'] else 2,whole_connected=len(arrived) if arm=='zero' else sum(r['connected'] for r in selected),prefix_steps_mean=statistics.mean(r['steps'] for r in pr if r['arm']==producer and r['arrived']) if arrived else None,total_steps_mean=statistics.mean(r['total_attempts'] for r in selected) if selected else None,coefficient_fits=3 if arm=='F' else 0,old_training_bank_and_MLP_shared=True))
    for name,data in [('per_output',rows),('summary',aggregates),('paired_contrasts',summary['decisions']),('fit_means',fitmeans),('prefixes',pr)]:write_csv(P/f'tables/additional_confirmation_{ds}_{name}.csv',data)
    labels={'F':r'$L_{\mathrm{full}}$','Q':'Q','N':'N','NS':'NS','zero':'Zero'}
    for mode in ['shared','full']:
        lines=[r'\begin{table}[ht]',r'\centering\small',r'\caption{'+ds.upper()+r' frozen new-bank/new-start confirmation, '+('shared C prefixes.' if mode=='shared' else 'complete policies.')+r' Fixed 2,048 suffix opportunities. Learned/Q SDs use three bank means over the two planned starts; other SDs use starts. Banks and starts are fresh, while old examples and the original MLP are shared. Missing arrivals remain in the planned denominator.}',r'\label{tab:additional-confirmation-'+ds+'-'+mode+'}',r'\begin{tabular}{lrrrr}',r'\toprule',r'Policy & Eligible/planned & Error & SD & Connected \\',r'\midrule']
        for r in aggregates:
            if (mode=='shared' and r['producer']!='C') or (mode=='full' and r['arm']=='zero'):continue
            value='--' if r['mean'] is None else f"{r['mean']:.6f}";sd='--' if r['sample_sd'] is None else f"{r['sample_sd']:.6f}"
            label=labels[r['arm']] if mode=='shared' else r['producer']+'+'+labels[r['arm']]
            lines.append(f"{label} & {r['eligible']}/{r['planned']} & {value} & {sd} & {r['whole_connected']}"+r' \\')
        lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}'];(P/f'tables/additional_confirmation_{ds}_{mode}.tex').write_text('\n'.join(lines)+'\n')
    receipt=dict(source_files={str(p):sha(p) for p in [source/'manifest.json',source/'acceptance.json',source/'summary.json',source/'zero.json',BASE/f'prefix/{ds}/summary.json']},dataset=ds,core_gate_passed=summary['passed'],decisions=summary['decisions'],aggregates=aggregates,readouts=len(rows),chains=len(final),new_experiments=0)
    (P/f'additional_confirmation_{ds}_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(ds,'confirmation exported',summary['passed'],len(final))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset',choices=['acm','freebase']);main(p.parse_args().dataset)
