"""Review-only manifest CLI. It has no formal launcher or worker import."""
import argparse,json,pathlib
ROOT=pathlib.Path(__file__).resolve().parent
def plan():
    cfg=json.loads((ROOT/'protocol.json').read_text());trains=[];samples=[]
    for seed in cfg['train_seeds']:
        task=f'PubMed_s{seed}_O'
        trains.append(dict(id=task,train_seed=seed,updates=1000,readouts=cfg['final_readouts_per_trajectory'],directory=task))
        for readout in cfg['final_readouts_per_trajectory']:
            for sample in cfg['sample_seeds']:
                samples.append(dict(id=f'{task}_{readout}_{sample}',depends_on=task,readout=readout,
                                    sample_seed=sample,T=1000,directory=f'{task}/{readout}/sample_{sample}'))
    return dict(status='review_only_not_activated',training=trains,sampling=samples,formal_authorized=False)
def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['plan','formal']);a=p.parse_args()
    if a.command=='formal':raise SystemExit('Formal execution is not authorized or wired in this minimal review package.')
    print(json.dumps(plan(),indent=2))
if __name__=='__main__':main()
