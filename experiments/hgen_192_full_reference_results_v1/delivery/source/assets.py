"""Fit the single fixed Hin2Vec asset per reference, following linked script."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,time,traceback,contextlib
import numpy as np,torch
from torch.utils.data import DataLoader,TensorDataset
from common import *
torch.set_num_threads(1)
def main(d):
 folder=OUT/'data'/d;mapping=read(folder/'hin2vec_mapping.json');z=np.load(str(folder/'hin2vec_examples.npz'));check_time(1200)
 seed(PROTOCOL['embedding_seeds'][d]);m=module(ROOT/'vendor/hin2vec/model.py','frozen_hin2vec');h=m.HIN2vec(192,mapping['path_size'],32,True).cuda();opt=torch.optim.AdamW(h.parameters());ds=TensorDataset(torch.from_numpy(z['x']),torch.from_numpy(z['y']));dl=DataLoader(ds,batch_size=32,shuffle=True,num_workers=0)
 if (folder/'embedding.npy').exists():raise FileExistsError('do not refit a fixed asset')
 torch.save(dict(model=h.state_dict(),optimizer=opt.state_dict(),rng=rng()),str(folder/'hin2vec_initial.pt'));begin=time.time()
 # Literal upstream train(), no changed negative sampling, optimizer or epochs.
 with (folder/'hin2vec_training.log').open('w') as log:
  for epoch in range(4):
   check_time(900)
   with contextlib.redirect_stdout(log):m.train(200,h,'cuda',dl,opt,torch.nn.BCELoss(),epoch)
   log.flush();save(folder/'asset_progress.json',dict(stage='Hin2Vec',epochs=epoch+1,seconds=time.time()-begin))
 emb=h.start_embeds.weight.detach().cpu().numpy()[[mapping['node2id'][str(i)] for i in range(192)]]
 assert emb.shape==(192,32) and np.isfinite(emb).all()
 np.save(str(folder/'embedding.npy'),emb);torch.save(dict(model=h.state_dict(),optimizer=opt.state_dict(),rng=rng(),epochs=4),str(folder/'hin2vec_final.pt'))
 save(folder/'embedding_receipt.json',dict(status='complete',dataset=d,seed=PROTOCOL['embedding_seeds'][d],epochs=4,updates=4*len(dl),examples=len(ds),seconds=time.time()-begin,sha256=sha(folder/'embedding.npy'),node_mapping_sha256=sha(folder/'hin2vec_mapping.json'),reference_sha256=sha(folder/'reference.npz'),native_source_sha256=sha(ROOT/'vendor/hin2vec/model.py'),only_current_reference=True,fixed_across_four_GAN_seeds=True))
 print(d,'Hin2Vec fixed asset complete',time.time()-begin)
if __name__=='__main__':
 try:main(sys.argv[1])
 except BaseException:save(OUT/('asset_failure_'+sys.argv[1]+'.json'),dict(traceback=traceback.format_exc(),epoch=time.time()));raise
