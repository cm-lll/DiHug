"""Explicit local CSV export. Does not load checkpoints, models or contact hosts."""
import argparse,csv,json
from pathlib import Path
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    source=a.run.resolve();dest=a.output.resolve()
    if not source.is_dir():raise FileNotFoundError(source)
    if source==dest or source in dest.parents or dest in source.parents:raise ValueError('new export must be separate from run')
    dest.mkdir(parents=True,exist_ok=False)
    with (dest/'training.csv').open('w',newline='') as h:
        writer=csv.writer(h);writer.writerow(['train_seed','updates','arm','loss','gradient_norm','update_seconds','elapsed_seconds'])
        for folder in sorted(source.glob('seed*')):
            if not (folder/'training.jsonl').is_file():continue
            for line in (folder/'training.jsonl').read_text().splitlines():
                x=json.loads(line)
                for arm,v in x['arms'].items():writer.writerow([x['seed'],x['completed_updates'],arm,v['loss'],v['gradient_norm'],v['update_seconds'],x['elapsed_seconds']])
    with (dest/'validation.csv').open('w',newline='') as h:
        writer=csv.writer(h);writer.writerow(['train_seed','updates','arm','t','CE','constant_prior_CE'])
        for folder in sorted(source.glob('seed*')):
            for path in sorted(folder.glob('validation_*.json')):
                for arm,value in json.loads(path.read_text()).items():
                    for row in value['rows']:writer.writerow([folder.name.removeprefix('seed'),int(path.stem.split('_')[-1]),arm,row['t'],row['CE'],row['constant_prior_CE']])
    (dest/'scope.json').write_text(json.dumps({'input':str(source),'checkpoints_and_raw_graphs_stay_in_original_run':True,'no_model_calls':True},indent=2)+'\n')
    print(dest)
if __name__=='__main__':main()
