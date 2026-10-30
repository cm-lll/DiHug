import pathlib,json,hashlib,os,random,time,ast,importlib.util,collections
ROOT=pathlib.Path(__file__).resolve().parent
PROJECT=pathlib.Path('/data2/lyh/gnn_project/DiHug')
OUT=PROJECT/'results/hgen_192_full_reference_20261030T091112'
PROTOCOL=json.loads((ROOT/'protocol.json').read_text())
def read(p):return json.loads(pathlib.Path(p).read_text())
def save(p,x):
 p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False));os.replace(str(tmp),str(p))
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def module(p,name):
 s=importlib.util.spec_from_file_location(name,str(p));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def check_time(reserve=0):
 if time.time()>read(OUT/'ledger.json')['science_stop_epoch']-reserve:raise TimeoutError('bounded science stop')
def source_manifest():return {str(p.relative_to(ROOT)):sha(p) for p in sorted(ROOT.rglob('*')) if p.is_file() and p.suffix in ('.py','.json') and '__pycache__' not in p.parts}
def seed(s):
 import numpy as np,torch
 random.seed(s);np.random.seed(s);torch.manual_seed(s)
 if torch.cuda.is_available():torch.cuda.manual_seed_all(s)
def load_core(instrument=False):
 import numpy as np,scipy.sparse as sp,torch
 from torch import nn
 from torch.nn import functional as F
 from torch.autograd import Variable
 from types import SimpleNamespace
 env=dict(np=np,sp=sp,torch=torch,nn=nn,F=F,Variable=Variable,random=random,Counter=collections.Counter,__name__='isolated_author_HGEN')
 unames={'make_noise','delete_from_tail','pad_along_axis','score_matrix_from_random_walks','heterogeneous_graph_assemble','sample_from_dict','array_subtraction','symmetric'}
 nodes=[n for n in ast.parse((ROOT/'vendor/hgen/utils.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name in unames]
 tree=ast.Module(body=nodes)
 if hasattr(ast,'TypeIgnore'):tree.type_ignores=[]
 exec(compile(tree,str(ROOT/'vendor/hgen/utils.py'),'exec'),env);env['utils']=SimpleNamespace(**{k:env[k] for k in unames})
 names={'Generator','Discriminator','train','frequent_meta_path_pattern','meta_path_frequency'}
 nodes=[n for n in ast.parse((ROOT/'vendor/hgen/hgen.py').read_text()).body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
 if instrument:
  tr=next(n for n in nodes if n.name=='train');loop=tr.body[-1].body[0];assert isinstance(loop,ast.For)
  loop.body.append(ast.parse('record_update(batches_done, loss_D)').body[0])
 tree=ast.Module(body=nodes)
 if hasattr(ast,'TypeIgnore'):tree.type_ignores=[]
 exec(compile(ast.fix_missing_locations(tree),str(ROOT/'vendor/hgen/hgen.py'),'exec'),env)
 return env
def setup(env,types,embedding,device,epochs=5):
 import numpy as np,torch
 from types import SimpleNamespace
 groups={i:np.flatnonzero(types==i).tolist() for i in range(4)}
 a=SimpleNamespace(batch_size=32,noise_dim=64,noise_type='Uniform',hidden_units=128,num_G_layer=5,max_path_len=4,W_down_generator_size=128,W_down_discriminator_size=128,neighbor_sampling_size=120,dataset='IMDB_movie',lr_gen=1e-4,lr_dis=1e-4,n_critic=5,clip_value=.03,n_epochs=epochs)
 emb=[torch.tensor(embedding[groups[i]],dtype=torch.float32,device=device) for i in range(4)]
 g=env['Generator'](groups,len(types),emb,torch.device(device),a).to(device);d=env['Discriminator'](len(types),torch.device(device),a).to(device)
 go=torch.optim.RMSprop(g.parameters(),lr=1e-4);do=torch.optim.RMSprop(d.parameters(),lr=1e-4)
 env.update(args=a,device=torch.device(device),generator=g,discriminator=d,optimizer_G=go,optimizer_D=do)
 return g,d,go,do,groups
def rng():
 import numpy as np,torch
 return dict(py=random.getstate(),np=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
def restore_rng(r):
 import numpy as np,torch
 random.setstate(r['py']);np.random.set_state(r['np']);torch.set_rng_state(r['torch'])
 if r['cuda']:torch.cuda.set_rng_state_all(r['cuda'])
def cell(d,s):return d+'_s'+str(s)
