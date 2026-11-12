"""CPU synthetic integration only; tiny T1000 chains, no user data or GPU."""
import hashlib,json,os,subprocess,sys,tempfile
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
def main():
    with tempfile.TemporaryDirectory(prefix='dihug_server_test_') as temp:
        p=Path(temp);np.savez(p/'toy.npz',types=np.array([0,0,0,1,1,1]),edges=np.array([[0,1,2,3,4],[1,2,3,4,5]]),schema=np.array([[0,0],[0,1],[1,1]]))
        cfg=json.loads((HERE/'configs/IMDB_A.json').read_text());cfg['model'].update(layers=1,width=32,edge_width=16,global_width=16,heads=4,ff_node=32,ff_edge=16,ff_global=16,cn_hidden=8,joint_width=16)
        (p/'tiny.json').write_text(json.dumps(cfg));env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
        def run(mode,name,arm='A',extra=()):
            subprocess.run([sys.executable,str(HERE/'runtime.py'),mode,'--data',str(p/'toy.npz'),'--config',str(p/'tiny.json'),'--out',str(p/name),'--arm',arm,'--seed','7','--rounds','2','--device','cpu','--minutes','3',*extra],env=env,check=True)
        for arm in ['O','A']:run('smoke','smoke_'+arm,arm)
        run('train-ce','ce',extra=['--updates','1'])
        checkpoint=p/'ce/final_complete.pt';h=hashlib.sha256(checkpoint.read_bytes()).hexdigest();parent=['--checkpoint',str(checkpoint),'--checkpoint-sha256',h]
        run('train-ce','resumed',extra=[*parent,'--resume','--updates','2'])
        run('train-ce','continuous',extra=['--updates','2'])
        import torch
        sys.path.insert(0,str(HERE));import runtime
        a=torch.load(p/'resumed/final_complete.pt',map_location='cpu',weights_only=False);b=torch.load(p/'continuous/final_complete.pt',map_location='cpu',weights_only=False)
        assert runtime.s.state_digest(a)==runtime.s.state_digest(b),'CPU update/resume differs'
        run('sample','sample',extra=[*parent,'--readout','ema','--sample-seeds','31'])
        run('feedback','feedback',extra=[*parent,'--readout','ema','--updates','1'])
        print(json.dumps(dict(status='passed',CPU_synthetic_only=True,normal_O_A=True,continuous_vs_resume_full_state=True,EMA_sampling_T1000=True,two_terminal_training_chains=True,feedback_update=True,no_real_graph_or_GPU=True)))
if __name__=='__main__':main()
