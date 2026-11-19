"""Separate end-point and probability arithmetic checks, with explicit limits."""
from r2_common import *
import gzip,argparse,itertools
from scipy.special import logsumexp
from dihug.core.mixed_edit_diffusion import load_mixed_edit_checkpoint,mixed_edit_candidate_matrix,EditProposal
from dihug.core.sampling_session import read_snapshot,digest,restore_graph
from review_endpoints import parse,motifs

def training(run):
 reports=[]
 for ds in ASSETS:
  orders={}
  for seed in (210,211,212):
   initials={}
   for arm in ARMS:
    p=run/'fits'/ds/f'seed{seed}'/arm;r=json.loads((p/'report.json').read_text());z=np.load(p/'validation.npz')
    x,y,ids,_=load_data(ds,arm);val=ids==ids.max();assert np.array_equal(z['indices'],np.flatnonzero(val)) and np.array_equal(z['targets'],y[val])
    logits=z['logits'].astype(np.float64);logp=logits-logsumexp(logits,axis=1,keepdims=True);ce=float(-logp[np.arange(len(logp)),z['targets']].mean())
    assert abs(ce-r['validation']['ce'])<1e-6
    assert np.max(np.abs(np.exp(logp)-z['probabilities']))<1e-6
    assert r['updates']==ASSETS[ds]['epochs']*int(np.ceil((~val).sum()/64))
    m,_,_=load_mixed_edit_checkpoint(p/'mixed_edit_diffusion.pt');check,_,prob=eval_model(m,x[val],y[val],'cpu')
    assert np.max(np.abs(prob-z['probabilities']))<2e-5
    if arm!='M':assert torch.count_nonzero(m.network[0].weight[:,schema(ds)['zero_columns']])==0
    assert sha(p/'mixed_edit_diffusion.pt')==r['checkpoint_sha256']
    orders.setdefault(seed,set()).add(r['batch_order_sha256']);initials[arm]=torch.load(p/'initial.pt',map_location='cpu',weights_only=False)['state_dict']
    reports.append({'dataset':ds,'seed':seed,'arm':arm,'ce_independent':ce,'cpu_probability_max_error':float(np.max(np.abs(prob-z['probabilities']))),'updates':r['updates']})
   for key in initials['M']:
    a=initials['M'][key].clone()
    if key=='network.0.weight':a[:,schema(ds)['zero_columns']]=0
    assert torch.equal(a,initials['D'][key])
  assert all(len(x)==1 for x in orders.values())
 result={'passed':True,'fits':len(reports),'updates':sum(r['updates'] for r in reports),'records':reports,'scope':'all saved validation probabilities independently recomputed and CPU re-forwarded; paired initialization/order checked; no exhaustive training-gradient replay','scientific_winner':None}
 atomic(run/'training_acceptance.json',result);return result

def generated(run):
 manifest=json.loads((HISTORY/'manifest.json').read_text());public=json.loads((HISTORY/'public_motif_counts.json').read_text());records=[];values={}
 for ds in ASSETS:
  reference=parse(json.loads(Path(manifest['references'][ds]['clean_reference_path']).read_text()));templates=public[ds]['templates'];refcounts=motifs(reference,templates);assert refcounts==public[ds]['counts'];values[ds]={}
  for seed in (210,211,212):
   values[ds][seed]={}
   for arm in ARMS:
    loss=[]
    for prefix in range(100,108):
     p=run/'generation'/ds/f'train{seed}'/arm/f'prefix{prefix}';bind=json.loads((p/'binding.json').read_text());rep=json.loads((p/'report.json').read_text());final=json.loads((p/'final_graph.json').read_text())
     assert rep['completed'] and rep['suffix_attempts']==rep['expected_suffix_attempts'] and sha(p/'final_graph.json')==rep['final_sha256']
     source=Path(bind['source_transition']);assert sha(source)==bind['source_transition_sha256'];snap=read_snapshot(source);g=restore_graph(snap['graph']);base=parse(snap['graph']);chain='';count=0;scored=0
     checkpoint=Path(bind['training_checkpoint']);assert sha(checkpoint)==bind['checkpoint_sha256'];m,_,_=load_mixed_edit_checkpoint(checkpoint);m.eval()
     edges={f:set(map(tuple,es)) for f,es in g.family_edges.items()}
     with gzip.open(p/'actions.jsonl.gz','rt') as f:
      for line in f:
       item=json.loads(line);r=item['record'];chain=digest({'previous':chain,'record':r});assert chain==item['chain'];count+=1
       assert r['attempt']==snap['completed']+count
       assert abs(r['time']-(1-(r['attempt']-1)/20000))<1e-12
       if 'candidates' in r:
        assert digest(r['candidates'])==r['candidates_sha256'];cs=[EditProposal(a['action'],a['family'],tuple(map(tuple,a['old'])),tuple(map(tuple,a['new']))) for a in r['candidates']]
        x=mixed_edit_candidate_matrix(g,cs,r['time'],'typed_closure_compact_v2')
        with torch.no_grad():scores=m(torch.from_numpy(x)[None]).squeeze(0).numpy()
        assert np.max(np.abs(scores-np.array(r['scores'])))<2e-5 and int(scores.argmax())==r['selected_index']
        assert r['candidates'][r['selected_index']]==r['selected'];scored+=1
       a=r['selected'];family=a['family'];old=list(map(tuple,a['old']));new=list(map(tuple,a['new']));assert a['action'] in ('noop','switch')
       balance={}
       for u,v in old:
        assert (u,v) in edges[family];edges[family].remove((u,v));g.remove_edge(family,(u,v))
        for n in (u,v):balance[n]=balance.get(n,0)-1
       for u,v in new:
        assert u!=v and (u,v) not in edges[family];edges[family].add((u,v));g.add_edge(family,(u,v))
        for n in (u,v):balance[n]=balance.get(n,0)+1
       assert not any(balance.values())
     assert count==rep['suffix_attempts'] and chain==rep['action_chain']
     assert all(edges[f['name']]==set(map(tuple,f['edges'])) for f in final['families'])
     z=parse(final);assert np.array_equal(z['degree'],base['degree']) and z['edges']=={f:set(tuple(sorted(e)) for e in es) for f,es in edges.items()}
     counts=motifs(z,templates);error=sum(abs(a-b)/max(1,b) for a,b in zip(counts,refcounts))/len(refcounts);loss.append(error)
     e=set().union(*z['edges'].values());re=set().union(*reference['edges'].values())
     records.append({'dataset':ds,'training_seed':seed,'arm':arm,'prefix':prefix,'motifs':counts,'motif_loss':error,'components':z['components'],'LCC':z['LCC'],'N':z['n'],'M':z['m'],'reference_edge_jaccard_distance':1-len(e&re)/max(1,len(e|re)),'scores_recomputed_steps':scored,'all_selected_actions_replayed':count})
    values[ds][seed][arm]=loss
 rng=np.random.default_rng(20261015);contrasts={};passes=[]
 for ds in ASSETS:
  contrasts[ds]={}
  for c in ('S','D'):
   delta=np.array([np.array(values[ds][s][c])-np.array(values[ds][s]['M']) for s in (210,211,212)])
   # Crossed seed and common-start resampling, not 24 independent graphs.
   ts=rng.integers(3,size=(20000,3));ps=rng.integers(8,size=(20000,8));boot=delta[ts[:,:,None],ps[:,None,:]].mean((1,2));lo,hi=np.quantile(boot,[.00625,.99375]);passed=bool(lo>0 and (delta.mean(1)>0).all());passes.append(passed)
   contrasts[ds][c]={'control_minus_M_mean':float(delta.mean()),'interval_98_75':[float(lo),float(hi)],'per_training_seed':delta.mean(1).tolist(),'supports_M_increment':passed}
 # No external utility or topology isomorphism certification is claimed here.
 random_controls={}
 for ds in ASSETS:
  losses=[]
  reference=parse(json.loads(Path(manifest['references'][ds]['clean_reference_path']).read_text()));templates=public[ds]['templates'];refcounts=motifs(reference,templates)
  for prefix in range(100,108):
   p=HISTORY/ds/f'seed{prefix}/r0/R/final_graph.json';z=parse(json.loads(p.read_text()));counts=motifs(z,templates)
   losses.append(sum(abs(a-b)/max(1,b) for a,b in zip(counts,refcounts))/len(refcounts))
  random_controls[ds]={'source':'R1-verified historical R, same starts, repeat0, unchanged checkpoints not needed for R','outputs':8,'losses':losses,'mean':float(np.mean(losses)),
   'R_minus_new_M_by_training_seed':[float(np.mean(np.array(losses)-values[ds][s]['M'])) for s in (210,211,212)],'inference':'descriptive secondary contrast; reusing R does not create 24 independent random outputs'}
 result={'engineering_passed':True,'outputs':len(records),'joint_increment_supported':all(passes),'contrasts':contrasts,'historical_same_start_R':random_controls,'records':records,'limitations':['conditional stage2 only','two fixed references and fixed cached training trajectories','three training seeds give limited interval precision','motif not independent application utility','Jaccard is labeled change, not anonymous topological diversity','complete pipeline no-learning control still required; historical conditional R reused explicitly']}
 atomic(run/'generation_acceptance.json',result);return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--stage',choices=['training','generation'],required=True);a=p.parse_args();torch.set_num_threads(1);verify_manifest(a.run)
 r=training(a.run) if a.stage=='training' else generated(a.run);print(json.dumps({k:v for k,v in r.items() if k!='records'}),flush=True);verify_manifest(a.run)
