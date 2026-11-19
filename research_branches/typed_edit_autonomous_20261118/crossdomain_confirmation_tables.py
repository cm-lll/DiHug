"""Export every accepted fresh-start augmentation contrast without pooling pilots."""
import argparse,csv,json,statistics
from pathlib import Path
import crossdomain_augmentation_fit as repair
from round01 import b,ROOT
BASE=repair.BASE;P=ROOT/'results/original_paper_revision_20261118_potential'
def csvwrite(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True);fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def main(ds):
    source=BASE/f'confirmation/deploy/{ds}';repair.original.source_check(source);s=b.read(source/'summary.json');a=b.read(source/'acceptance.json');zero=b.read(source/'zero.json');prefix=BASE/f'confirmation/prefix/{ds}';pr=b.read(prefix/'summary.json')['rows']
    final=[r for r in s['rows'] if r['steps']==2048];out=BASE/f'confirmation/report/{ds}';out.mkdir(parents=True,exist_ok=False);aggregates=[];fitmeans=[]
    for producer,arms in [('C',['zero','T','U','B','Q','N','NS']),('D',['N'])]:
        for arm in arms:
            selected=[r for r in final if (r['producer'],r['arm'])==(producer,arm)]
            if arm=='zero':values=[zero[f"{r['seed']}_C"]['primary_error'] for r in pr if r['arm']=='C' and r['arrived']]
            elif arm in ['T','U','B','Q']:
                values=[]
                for bank in [2,3,4]:
                    rr=[r for r in selected if r['bank']==bank]
                    if rr:
                        v=statistics.mean(r['primary_error'] for r in rr);values.append(v);fitmeans.append(dict(dataset=ds,arm=arm,bank=bank,starts=len(rr),mean=v))
            else:values=[r['primary_error'] for r in selected]
            aggregates.append(dict(dataset=ds,producer=producer,arm=arm,mean=statistics.mean(values) if values else None,sample_sd=statistics.stdev(values) if len(values)>1 else None,triangle_mean=statistics.mean(r['triangle_vector_relative_l1'] for r in selected) if selected else None,cycle_mean=statistics.mean(r['cycle_vector_relative_l1'] for r in selected) if selected else None,sd_unit='three paired-bank means over eligible starts' if arm in ['T','U','B','Q'] else 'two starts',eligible=len(values) if arm=='zero' else len(selected),planned=6 if arm in ['T','U','B','Q'] else 2,connected=len(values) if arm=='zero' else sum(r['connected'] for r in selected),prefix_steps_mean=statistics.mean(r['steps'] for r in pr if r['arm']==producer),total_steps_mean=statistics.mean(r['total_attempts'] for r in selected) if selected else None))
    datasets=[('per_output',s['rows']),('paired_contrasts',s['decisions']),('summary',aggregates),('fit_means',fitmeans),('prefixes',pr)]
    for name,rows in datasets:
        csvwrite(out/f'{name}.csv',rows);csvwrite(P/f'tables/crossdomain_confirmation_{ds}_{name}.csv',rows)
    labels={'T':r'$L_{\mathrm{target}}$','U':r'$L_{\mathrm{ordinary}}$','B':r'$L_{\mathrm{base}}$','Q':'Q','N':'N','NS':'NS','zero':'Zero'}
    for mode in ['shared','full']:
        lines=[r'\begin{table}[ht]',r'\centering\small',r'\caption{'+ds.capitalize()+r' augmentation confirmation on fresh starts 17001/17002. '+('All selectors share each C prefix.' if mode=='shared' else 'Complete policies; C/D calibration differs.')+r' Targeted and ordinary banks each add 256 true-inverse lists per base bank. Normalization and Q remain fixed. Learned/Q SDs use three paired-bank means over the two starts; other SDs use starts. Every suffix uses 2,048 opportunities.}',r'\label{tab:crossdomain-'+ds+'-'+mode+'}',r'\begin{tabular}{lrrrr}',r'\toprule',r'Policy & Eligible/planned & Error & SD & Connected \\',r'\midrule']
        for r in aggregates:
            if (mode=='shared' and r['producer']!='C') or (mode=='full' and r['arm']=='zero'):continue
            label=labels[r['arm']] if mode=='shared' else r['producer']+'+'+labels[r['arm']];mean='--' if r['mean'] is None else f"{r['mean']:.6f}";sd='--' if r['sample_sd'] is None else f"{r['sample_sd']:.6f}"
            lines.append(f"{label} & {r['eligible']}/{r['planned']} & {mean} & {sd} & {r['connected']}"+r' \\')
        lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}'];text='\n'.join(lines)+'\n';(out/f'{mode}.tex').write_text(text);(P/f'tables/crossdomain_confirmation_{ds}_{mode}.tex').write_text(text)
    inputs=[Path(__file__),source/'manifest.json',source/'acceptance.json',source/'summary.json',source/'zero.json',prefix/'summary.json']
    receipt=dict(inputs={str(p):b.sha(p) for p in inputs},dataset=ds,implementation_accepted=a['passed'],core_gate_passed=s['passed'],decisions=s['decisions'],aggregates=aggregates,chains=len(final),readouts=len(s['rows']),new_fits=0,new_graphs=0,old_training_and_MLP_shared=True)
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');(P/f'crossdomain_confirmation_{ds}_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(dict(aggregates=aggregates,decisions=s['decisions']),indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset',choices=['pubmed','acm']);main(p.parse_args().dataset)
