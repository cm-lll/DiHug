"""Remove stale engineering-limit metadata before activation, without tensor edits."""
from common_run import OUT,read,save,sha,cfg_for,now
from prepare import load_case
from probe import equal
import torch
def main():
 tasks=read(OUT/'tasks.json');banks=read(OUT/'initial_banks.json');records=[]
 for j in tasks:
  _,base=load_case(j['dataset']);cfg=cfg_for(base,j['train_seed']);bank=banks[j['task_id']];p=__import__('pathlib').Path(bank['file']);z=torch.load(p,map_location='cpu');old=z['config'];changes={k:dict(old=old.get(k),new=v) for k,v in cfg.items() if old.get(k)!=v}
  assert set(changes)<={'max_updates_per_dataset_arm','runtime','source_note'}
  for k in ['model','optimizer','T','rounds','precision','inputs_per_update','datasets']:assert equal(old[k],cfg[k])
  oldsha=sha(p);oldpath=p.with_name(p.stem+'_pre_metadata_revision.pt');p.rename(oldpath);z['config']=cfg;torch.save(z,p)
  assert {k:__import__('hashlib').sha256(v.cpu().numpy().tobytes()).hexdigest() for k,v in z['model'].items()}==bank['tensor_sha256']
  bank['sha256']=sha(p);j['config']=cfg;records.append(dict(task=j['task_id'],changes=changes,old_file=str(oldpath),old_sha256=oldsha,new_sha256=bank['sha256'],all_model_tensors_unchanged=True))
 save(OUT/'initial_banks.json',banks);save(OUT/'tasks.json',tasks);save(OUT/'prepared/config_metadata_correction.json',dict(at=now(),purpose='remove unused historical five-update engineering limit before formal binding',scientific_fields_unchanged=True,records=records))
if __name__=='__main__':main()
