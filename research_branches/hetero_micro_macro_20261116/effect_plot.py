"""CPU figure: model means and within-model sampling SD, no inferential intervals."""
import os
import tempfile
os.environ['CUDA_VISIBLE_DEVICES'] = ''
cache = tempfile.TemporaryDirectory(prefix='micro_macro_plot_')
os.environ['MPLCONFIGDIR'] = cache.name
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(out):
    rows = json.loads((out / 'model_summary.json').read_text())
    arms = ['parent', 'CE', 'calibrated_micro_macro']
    metrics = [('degree_W1', 'Relation degree W1'), ('joint_relation_degree_L1', 'Joint relation degree L1'),
               ('triangle_abs_error', 'Absolute triangle count error'), ('MP3_TV', 'MP3 composition TV')]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)
    for di, dataset in enumerate(['IMDB', 'PubMed']):
        seeds = sorted(set(r['train_seed'] for r in rows if r['dataset']==dataset))
        for mi, (metric, title) in enumerate(metrics):
            ax = axes[di, mi]
            for seed in seeds:
                chosen = [next((r for r in rows if r['dataset']==dataset and r['train_seed']==seed and r['arm']==a and r['view']=='whole'), None) for a in arms]
                if not all(r and r['samples']==4 for r in chosen):
                    continue
                ax.errorbar(range(3), [r[metric+'_mean'] for r in chosen],
                            yerr=[r[metric+'_sample_SD'] for r in chosen], marker='o', capsize=3,
                            label=str(seed))
            ax.set_xticks(range(3), ['Parent', 'CE +128', 'Candidate +128'])
            ax.set_title(dataset + ': ' + title)
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
    fig.suptitle('Fixed EMA endpoint; 4 paired samples/model. Bars: sampling SD, not confidence intervals. Lower is better.', fontsize=11)
    fig.savefig(out / 'paired_quality.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    try:
        main(Path(sys.argv[1]))
    finally:
        cache.cleanup()
