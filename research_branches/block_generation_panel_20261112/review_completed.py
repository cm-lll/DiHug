"""Read-only CPU review of the two completed tickets, no model imports."""
import json,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def read(p):return json.loads(Path(p).read_text())
def main():
    b=ROOT/'results/block_generation_panel_20261112T015400'
    c=ROOT/'results/terminal_continuation_20261112T003000'
    assert read(b/'completion.json')['status']=='complete'
    assert read(c/'completion.json')['status']=='complete'
    assert not read(b/'verification.json')['missing'] and not read(c/'verification.json')['missing']
    wins=[];deltas=read(b/'paired_differences.json')
    for ds in ['IMDB','PubMed']:
        for r in [10,20]:
            rows=[x for x in deltas if x['dataset']==ds and x['rounds']==r and x['view']=='raw']
            assert len(rows)==4
            wins.append(dict(dataset=ds,rounds=r,paired_samples=4,descriptive_numerical_tie_tolerance=1e-12,metrics={k:dict(improved=sum(x[k]<-1e-12 for x in rows),tied=sum(abs(x[k])<=1e-12 for x in rows),worse=sum(x[k]>1e-12 for x in rows)) for k in ['degree_W1','joint_TV','triangle_abs_error','triangle_composition_TV','MP2_abs_error','MP2_TV','MP3_abs_error','MP3_TV','edge_abs_error']}))
    z=dict(status='reviewed_completed_no_new_compute',at=datetime.datetime.now().astimezone().isoformat(),
        block_panel=dict(graphs=24,new_graphs=20,reused_graphs=4,GPU_hours=read(b/'completion.json')['GPU_hours'],paired_directions=wins,
            conclusions=['IMDB R10 has degree and MP3 composition benefits, but MP2 count and triangle composition costs',
                         'IMDB R20 improves triangle count on average but has a large MP3 count outlier',
                         'PubMed blocked sampling worsens degree/MP3 composition in most paired samples; R20 reduces triangle count error on average with substantial other costs']),
        continuation_panel=dict(graphs=16,new_evaluation_graphs=8,reused_evaluation_graphs=8,GPU_hours=read(c/'completion.json')['GPU_hours'],
            conclusions=['IMDB update8 has mixed sample-specific degree/joint effects and improved mean MP3 error',
                         'PubMed update8 decreases mean triangle and MP3 count errors while worsening edge count, degree/joint and MP composition errors']),
        overlap_evaluation_graphs=4,unique_evaluation_graphs=36,
        limitations=['one training model per dataset', 'sampling SD is not training-seed SD',
                     'blocking panel uses fixed update2 raw weights, not update8',
                     'continuation uses two samples, blocking uses four; do not compare unpaired means as training trends',
                     'full-chain feedback remains a clean-head surrogate, not exact transition policy gradient',
                     'R1 versus blocked inference does not evaluate training with matching blocked context'],
        next_decision='Do not call blocking lossless or launch large-graph training. Bind feasible 64GiB block sizes and consistent training/sampling query protocol before a separate migration experiment.',
        resource_release='Both supervisors recorded own processes exited; host read-only check at 08:07 found all GPUs 0 MiB and no compute processes',new_training=0,new_generation=0)
    (b/'completion_review.json').write_text(json.dumps(z,indent=2,ensure_ascii=False))
    print(json.dumps(wins,indent=2))
if __name__=='__main__':main()
