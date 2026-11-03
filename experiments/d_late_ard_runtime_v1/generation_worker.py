"""One launcher-authorized fresh E-late model, 3000 updates and four chains.

Scientific update, EMA/checkpoint handling, validation and reverse sampling are
imported from the frozen runtime. This file changes identities and orchestration
only. The sole launcher owns process registration and GPU lease accounting.
"""
import argparse
import hashlib
import json
import os
import pickle
import resource
import signal
import socket
import time
import traceback
from common_run import ROOT, SCI, OUT, read, save, sha, now, check_time
import generation_adapter as adapter
import torch
from prepare import load_case
from model_factory import make_model, tensor_sha
from probe import rng_get, input_hash
from worker import atomic_checkpoint, validation
from integration import checkpoint_payload, ema_validation, final_readouts
from ema import ParameterEMA
from single_input import one_input, update
from sampling import sample_one
from boundary_transactions import TrainingBoundary


def setup():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if torch.cuda.device_count() != 1 or '2080 Ti' not in torch.cuda.get_device_name(0):
        raise RuntimeError('one approved2080Ti required')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticket', required=True)
    args = parser.parse_args()
    ticket = read(args.ticket)
    job = ticket['job']
    approval = read(OUT/'approval.json')
    if (approval.get('authorized') is not True or os.getppid() != approval['launcher_pid']
            or socket.gethostname() != approval['host']
            or os.environ.get('CUDA_VISIBLE_DEVICES') != str(job['gpu_slot'])):
        raise PermissionError('sole launcher/host/device identity failed')
    if (job['dataset'] != 'PubMed' or job['arm'] not in ('A','D','E') or job['method'] not in adapter.METHODS
            or job.get('reuse_completed_baseline') or job['updates'] != 3000
            or job['sample_seeds'] != list(adapter.SAMPLE_SEEDS)):
        raise ValueError('job outside frozen new-training panel')
    for filename, expected in approval['source_files'].items():
        if sha(filename) != expected:
            raise ValueError('source/config binding changed: '+filename)
    folder = OUT/'run'/job['task_id']
    folder.mkdir(parents=True, exist_ok=False)
    completed = chains = 0
    model = optimizer = ema = rng = training = None
    stopping = False
    status = dict(task_id=job['task_id'], dataset='PubMed', method=job['method'],
                  train_seed=job['train_seed'], GPU=job['gpu_slot'], GPU_UUID=ticket['GPU_UUID'],
                  pid=os.getpid(), started=now(), status='initializing')

    def stop(*args):
        nonlocal stopping
        stopping = True

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, stop)

    def check():
        if stopping:
            raise InterruptedError('own worker stop requested')
        check_time()

    def event(**values):
        status.update(values, completed_updates=completed, completed_chains=chains, last_event=now())
        save(folder/'progress.json', status)

    def snapshot(name):
        atomic_checkpoint(folder/name, training.checkpoint(binding))

    def stream_hashes():
        state = rng_get(rng, 'cuda')
        return {name: hashlib.sha256(value.cpu().numpy().tobytes() if torch.is_tensor(value)
                else pickle.dumps(value, protocol=4)).hexdigest() for name, value in state.items()}

    event()
    try:
        setup()
        check()
        seed, cfg, method = job['train_seed'], job['config'], job['method']
        adapter.verify_configuration(cfg, seed)
        case, _ = load_case('PubMed')
        dictionary = approval['dictionaries'][method]
        basis, dictionary_receipt = adapter.load_dictionary(dictionary)
        encoder_installation = adapter.install_dictionary_encoder(dictionary, basis)
        save(folder/'encoder_installation.json', encoder_installation)
        bank = approval['initial_banks'][job['task_id']]
        if sha(bank['file']) != bank['sha256']:
            raise ValueError('initial bank hash mismatch')
        z = torch.load(bank['file'], map_location='cpu')
        if (z['updates'] != 0 or z['train_seed'] != seed or z['dataset'] != 'PubMed'
                or z['arm'] != job['arm'] or z['method'] != method or z['config'] != cfg
                or z['dictionary_record'] != dictionary_receipt):
            raise ValueError('initial bank identity/dictionary mismatch')
        model = make_model(case, cfg, job['configuration'], basis).cuda()
        model.load_state_dict(z['model'], strict=True)
        if {k:tensor_sha(v) for k,v in model.state_dict().items()} != bank['tensor_sha256']:
            raise ValueError('actual initial tensors disagree')
        optimizer = torch.optim.AdamW(model.parameters(), **{k:v for k,v in cfg['optimizer'].items() if k != 'type'})
        ema = ParameterEMA(model)
        if optimizer.state or {id(p) for g in optimizer.param_groups for p in g['params']} != {id(p) for p in model.parameters() if p.requires_grad}:
            raise ValueError('fresh optimizer membership/state failed')
        rng, streams = adapter.reset_training(seed, 'cuda')
        if any(streams['state_sha256'][k] != v for k,v in z['CPU_training_rng']['state_sha256'].items() if k != 'device'):
            raise ValueError('paired main RNG reset mismatch')
        historical_streams = None
        historical_path = adapter.baseline_folder(seed)/'training.jsonl'
        if seed in (2026110221, 2026110222, 2026110321):
            historical_streams = [json.loads(line) for line in historical_path.read_text().splitlines()]
            if len(historical_streams) != 3000 or [r['update'] for r in historical_streams] != list(range(1,3001)):
                raise ValueError('archived full training-stream record incomplete')
        binding = dict(dataset='PubMed', train_seed=seed, arm=job['arm'], method=method, configuration=job['configuration'],
            config=cfg, updates=3000, inputs_per_update=1, initial_bank=bank,
            dictionary_record=dictionary_receipt, approval_sha256=sha(OUT/'approval.json'),
            encoder_installation=encoder_installation,
            source_manifest_sha256=sha(OUT/'source_manifest.json'),
            historical_training_streams_sha256=sha(historical_path) if historical_streams is not None else None)
        save(folder/'binding.json', binding)
        save(folder/'training_rng_reset.json', streams)
        save(folder/'initialization.json', dict(proof=z['initialization'], all_initial_tensors_exact=True,
             saved_before_first_forward=True, new_optimizer_empty=True, new_EMA_theta0=True, parameters=job['parameters']))
        training = TrainingBoundary(model, optimizer, ema, rng, 'cuda', completed)
        snapshot('initial.pt')
        del z
        from hsgd.bw_preflight.common import environment
        save(folder/'environment.json', environment('cuda'))
        validations = []

        def validate(position):
            validations.append(dict(updates=position, raw=validation(model,case,cfg,seed,rng),
                ema_0_99=ema_validation(validation,model,ema,case,cfg,seed,rng)))
            save(folder/'validation.json', validations)

        contracts = read(OUT/'prepared'/f'input_contract_PubMed_{seed}.json')
        if len(contracts) != 3000:
            raise ValueError('full input contract required')
        validate(0)
        event(status='training')
        for position in range(1,3001):
            check()
            with training.step(position):
                inp = one_input(case,cfg,position,rng)
                ih = input_hash([inp])
                expected = contracts[position-1]
                if ih != expected['sha256'] or inp['t'] != expected['t'] or len(inp['ids']) != expected['slots']:
                    raise ValueError('paired input mismatch before update '+str(position))
                began = time.perf_counter()
                before = stream_hashes()
                if historical_streams is not None and before != historical_streams[position-1]['training_streams_before']:
                    raise ValueError('archived common training streams differ before update '+str(position))
                rec = update(model,optimizer,case,inp,1000,'cuda',ema,completed,transaction=training)
            completed = training.completed
            after = stream_hashes()
            rec.update(update=completed,t=inp['t'],input_sha256=ih,paired_input_exact=True,
                training_streams_before=before,training_streams_after=after,at=now(),
                total_update_seconds=time.perf_counter()-began)
            with (folder/'training.jsonl').open('a') as f:
                f.write(json.dumps(rec)+'\n')
            if historical_streams is not None and after != historical_streams[position-1]['training_streams_after']:
                raise ValueError('archived common training streams differ after update '+str(position))
            if position % 5 == 0 or position == 1:
                event(status='training',last_loss=rec['loss'],last_update_seconds=rec['total_update_seconds'])
            if position in (1000,2000,3000):
                snapshot('final.pt' if position == 3000 else f'checkpoint_u{position}.pt')
                validate(position)
        states = final_readouts(model,ema,completed)
        checkpoint_hash = sha(folder/'final.pt')
        for kind, state in [('raw',states['raw']),('ema_0.99',states['ema'])]:
            atomic_checkpoint(folder/f'readout_{kind}.pt',dict(model=state,readout=kind,
                decay=.99 if kind != 'raw' else None,updates=3000,train_seed=seed,binding=binding,
                checkpoint_sha256=checkpoint_hash))
        del model,optimizer,ema,states
        model = optimizer = ema = training = None
        torch.cuda.empty_cache()
        row = dict(id=job['task_id'],dataset='PubMed',train_seed=seed,configuration=job['configuration'],
            readout_file=str(folder/'readout_ema_0.99.pt'),readout_sha256=sha(folder/'readout_ema_0.99.pt'),
            parent_approval_sha256=sha(OUT/'approval.json'))
        for ss in adapter.SAMPLE_SEEDS:
            check()
            event(status='sampling',sample_seed=ss)
            sample_one(row,ss,folder/'ema_0.99'/f'sample_{ss}',case,cfg,basis,OUT,check,event)
            chains += 1
        event(status='complete')
    except BaseException as exc:
        status.update(status='failed_or_incomplete',error=repr(exc),traceback=traceback.format_exc())
        if training is not None:
            completed = training.completed
            status['recovery'] = training.last_failure
        if model is not None and training is not None and training.resumable and training.phase == 'idle':
            try:
                snapshot(f'stopped_u{completed}.pt')
            except BaseException as save_error:
                status['checkpoint_save_error'] = repr(save_error)
        else:
            status['incomplete_update_not_presented_as_complete'] = True
            status['exact_resume_allowed'] = False
            status['use_previous_complete_checkpoint_only'] = True
        event()
    finally:
        status.update(completed_updates=completed,completed_chains=chains,finished=now(),
                      peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        if torch.cuda.is_initialized():
            status.update(peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                          peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
        save(folder/'summary.json',status)
    return 0 if status['status'] == 'complete' else 2


if __name__ == '__main__':
    raise SystemExit(main())
