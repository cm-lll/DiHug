"""Tables/plots from existing extracted records; no network or probability inference."""
import os
os.environ['MPLBACKEND']='Agg';os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,csv,json,pathlib,statistics
import matplotlib.pyplot as plt
from analyze import csvwrite,save,SEEDS
def load(p):return list(csv.DictReader(p.open()))
def mean(xs):
    if not xs or any(x is None for x in xs):return None
    return sum(xs)/len(xs)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);a=ap.parse_args();o=pathlib.Path(a.output);(o/'plots').mkdir(exist_ok=True)
    v=load(o/'validation_total.csv');f=load(o/'validation_relations.csv');tr=load(o/'trajectory_total.csv');fr=load(o/'trajectory_relations.csv');summary=[];means=[];intervals=[]
    for ds,seeds in SEEDS.items():
        fig,axes=plt.subplots(2,3,figsize=(15,8),sharex=True,sharey=True)
        ref=323 if ds=='IMDB' else 926
        for ax,arm in zip(axes.flat,'OABCDE'):
            ax.axhline(ref,color='black',ls=':',label=f'clean reference {ref}')
            for color,seed in zip(['tab:blue','tab:orange'],seeds):
                rows=[x for x in tr if x['dataset']==ds and int(x['train_seed'])==seed and x['arm']==arm]
                times=sorted({int(x['state_t']) for x in rows},reverse=True)
                for sample in [2026102821,2026102822]:
                    ys=[float(next(x for x in rows if int(x['state_t'])==t and int(x['sample_seed'])==sample)['actual_edges']) for t in times]
                    ax.plot([1000-t for t in times],ys,':',color=color,alpha=.35,lw=.7)
                ys=[mean([float(x['actual_edges']) for x in rows if int(x['state_t'])==t]) for t in times]
                ax.plot([1000-t for t in times],ys,'o--',color=color,ms=3,label=f'train {seed}; 2-chain mean')
                for t,y in zip(times,ys):means.append(dict(dataset=ds,train_seed=seed,arm=arm,state_t=t,reverse_steps_completed=1000-t,actual_edges_mean=y,reference_edges=ref,samples=2))
                for sample in [2026102821,2026102822]:
                    rr=sorted([x for x in rows if int(x['sample_seed'])==sample],key=lambda x:int(x['state_t']),reverse=True)
                    for x,y in zip(rr,rr[1:]):intervals.append(dict(dataset=ds,train_seed=seed,arm=arm,sample_seed=sample,input_saved_state_t=x['state_t'],later_saved_state_t=y['state_t'],earlier_actual_edges=x['actual_edges'],later_actual_edges=y['actual_edges'],observed_net_count_change=int(y['actual_edges'])-int(x['actual_edges']),meaning='difference of saved actual counts; not cumulative births/deaths or a q prediction'))
                val=[x for x in v if x['dataset']==ds and int(x['train_seed'])==seed and x['arm']==arm and x['updates']=='1000']
                row=dict(dataset=ds,train_seed=seed,arm=arm,reference_edges=ref,final_actual_edges_mean=ys[-1])
                for x in val:
                    t=x['noise_t']
                    for k in ['total_CE','p0_sum','p0_signed_bias','positive_CE_mean','negative_CE_mean']:row['ordinary_t'+t+'_'+k]=float(x[k])
                summary.append(row)
            ax.set_title(ds+' '+arm);ax.set_xlabel('completed reverse steps (1000 - state t)');ax.set_ylabel('actual binary edges');ax.legend(fontsize=7)
        fig.suptitle('Saved generated states only; dashed segments are not observed intermediate trajectories')
        fig.tight_layout();fig.savefig(o/'plots'/f'{ds}_actual_edges.png',dpi=160);plt.close(fig)
    csvwrite(o/'validation_vs_generation.csv',summary);csvwrite(o/'trajectory_two_sample_means.csv',means);csvwrite(o/'saved_interval_changes.csv',intervals)
    # All relations, O/D focus in PubMed. Individual points are retained in CSV.
    refschema=json.loads((o/'references/PubMed_schema.json').read_text())
    for arm in 'OD':
        fig,axes=plt.subplots(2,5,figsize=(18,7),sharex=True)
        for family,ax in enumerate(axes.flat):
            ax.axhline(refschema['reference_edges'][family],color='black',ls=':')
            for color,seed in zip(['tab:blue','tab:orange'],SEEDS['PubMed']):
                rows=[x for x in fr if x['dataset']=='PubMed' and x['arm']==arm and int(x['train_seed'])==seed and int(x['family'])==family];times=sorted({int(x['state_t']) for x in rows},reverse=True)
                ys=[mean([float(x['actual_edges']) for x in rows if int(x['state_t'])==t]) for t in times]
                ax.plot([1000-t for t in times],ys,'o--',ms=3,color=color,label=str(seed));ax.set_title(refschema['family_names'][family]);ax.set_xlabel('completed reverse steps');ax.legend(fontsize=7)
        fig.suptitle(f'PubMed {arm}: actual relation edges; 2-chain means, saved points only');fig.tight_layout();fig.savefig(o/'plots'/f'PubMed_{arm}_relations.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,4))
    for ax,seed in zip(axes,SEEDS['IMDB']):
        for arm in 'OABCDE':
            rows=sorted([x for x in f if x['dataset']=='IMDB' and int(x['train_seed'])==seed and x['arm']==arm and x['noise_t']=='1' and x['family']=='8'],key=lambda x:int(x['updates']))
            ax.plot([int(x['updates']) for x in rows],[float(x['p0_sum']) for x in rows],'o--',label=arm)
        ax.axhline(165,color='black',ls=':');ax.set_title(f'IMDB ordinary t=1, train {seed}');ax.set_xlabel('updates (only 0 / 500 / 1000 observed)');ax.set_ylabel('MG p0 sum, not actual sampled edges');ax.legend(ncol=3)
    fig.tight_layout();fig.savefig(o/'plots/IMDB_MG_validation.png',dpi=170);plt.close(fig)
    # Easy-to-read complete final-validation panel, no collapsed training seeds.
    text=['# 完整面板索引','', 'p0列均来自1000更新固定普通噪声验证；末图列是两条已有自由生成链实际边数均值。不同来源并列描述，不拼接为因果轨迹。','', '|数据集|训练seed|组|t=1 CE / p0和|t=500 CE / p0和|t=1000 CE / p0和|末图实际边数均值|','|---|---:|---|---:|---:|---:|---:|']
    for x in summary:text.append('|'+ '|'.join([x['dataset'],str(x['train_seed']),x['arm']]+[f"{x[f'ordinary_t{t}_total_CE']:.6f} / {x[f'ordinary_t{t}_p0_sum']:.2f}" for t in [1,500,1000]]+[f"{x['final_actual_edges_mean']:.1f}"])+'|')
    text+=['','所有0/500/1000更新、三噪声时刻的总表：[validation_total.csv](validation_total.csv)；分关系和正/负CE：[validation_relations.csv](validation_relations.csv)。','', '完整生成状态总表：[trajectory_total.csv](trajectory_total.csv)；分关系表：[trajectory_relations.csv](trajectory_relations.csv)。','', '## 已保存状态曲线','']
    for p in sorted((o/'plots').glob('*.png')):text.extend([f'![{p.stem}](plots/{p.name})',''])
    (o/'TABLES.md').write_text('\n'.join(text)+'\n')
    save(o/'render_receipt.json',dict(status='complete',comparison_cells=len(summary),intervals=len(intervals),figures=5,model_calls=0))
if __name__=='__main__':main()
