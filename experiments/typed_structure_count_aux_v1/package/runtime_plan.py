"""CPU-only task identities, immutable schedule, and prospective run binding.

Importing this module neither imports torch nor queries/activates a device.
"""
import hashlib
import json
import pathlib
import time

TRAIN_SEEDS = {'IMDB': (2026102811, 2026102834), 'PubMed': (2026102812, 2026102833)}
SAMPLE_SEEDS = (2026102821, 2026102822)
QUEUE = [('PubMed', 2026102812, 'E'), ('PubMed', 2026102833, 'E'),
         ('PubMed', 2026102812, 'D'), ('PubMed', 2026102833, 'D'),
         ('IMDB', 2026102811, 'E'), ('IMDB', 2026102834, 'E'),
         ('IMDB', 2026102811, 'D'), ('IMDB', 2026102834, 'D')]
BUDGET_SECONDS = 145 * 60


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def task_id(dataset, train_seed, arm):
    if dataset not in TRAIN_SEEDS or train_seed not in TRAIN_SEEDS[dataset] or arm not in ('D', 'E'):
        raise ValueError('outside frozen D/E task panel')
    return '%s_s%d_%s' % (dataset, train_seed, arm)


def validate_protocol(protocol):
    if protocol['train_seeds'] != {d: list(s) for d, s in TRAIN_SEEDS.items()}:
        raise ValueError('training seed panel changed')
    if protocol['sample_seeds'] != list(SAMPLE_SEEDS):
        raise ValueError('sampling seed panel changed')
    for key, value in dict(updates_per_model=1000, models=8, complete_chains=16, T=1000, rounds=1, S=18336).items():
        if protocol[key] != value:
            raise ValueError('frozen protocol differs: ' + key)
    if protocol['checkpoint_updates'] != [0, 500, 1000]:
        raise ValueError('checkpoint policy')
    if [(x['dataset'], x['seed'], x['arm']) for x in protocol['queue']] != QUEUE or [x['slot'] for x in protocol['queue']] != list(range(8)):
        raise ValueError('proposal queue changed')


def plan(protocol, output):
    validate_protocol(protocol)
    output = pathlib.Path(output).resolve()
    tasks = []
    for slot, (dataset, seed, arm) in enumerate(QUEUE):
        label = task_id(dataset, seed, arm)
        folder = output / label
        tasks.append(dict(task_id=label, slot=slot, dataset=dataset, train_seed=seed, arm=arm,
                          output=str(folder), updates=1000, checkpoint_updates=[0, 500, 1000],
                          initialization='experiment_contract.initialization', role_basis=(arm == 'E'),
                          samples=[dict(sample_seed=s, output=str(folder / ('sample_' + str(s))),
                                        dependency=label + ':final1000', T=1000,
                                        order=i, after='training' if i == 0 else 'sample_' + str(SAMPLE_SEEDS[i-1]))
                                   for i, s in enumerate(SAMPLE_SEEDS)]))
    return dict(mode='dry_run_only', formal_authorized=False, formal_start=None, formal_deadline=None,
                task_count=8, sample_count=16, tasks=tasks, optimizer_steps=0, GPU_calls=0,
                worker_launches=0, proposed_budget_seconds=BUDGET_SECONDS)


def assignment(cost, updates, devices):
    if updates != 1000 or len(devices) != 8 or len(set(devices)) != 8:
        raise ValueError('exactly 1000 updates and eight distinct devices required; no tier selection')
    if [(x['dataset'], x['seed'], x['arm']) for x in cost['queue']] != QUEUE or [x['slot'] for x in cost['queue']] != list(range(8)):
        raise ValueError('cost queue differs from execution queue')
    return [[t] for t in QUEUE], [x['estimated_worker_seconds'] for x in cost['queue']]


def source_binding(package_root):
    root = pathlib.Path(package_root).resolve().parent
    files = {}
    # Runtime/metric scripts, protocol, data, frozen dictionaries and historical
    # payload index are bound together. Receipts/output directories are excluded.
    for folder in ('package', 'dictionaries'):
        for p in (root / folder).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py', '.json', '.npz'):
                files[p.relative_to(root).as_posix()] = sha(p)
    for name in ('summarize.py', 'summary_metrics.py', 'proposal.json', 'history/index.json', 'evidence/execution/cost_proposal.json'):
        files[name] = sha(root / name)
    digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return files, digest


def check_activation(ledger, approval, approval_path, protocol_path, output=None):
    if ledger['approval_sha256'] != sha(approval_path) or ledger['protocol_sha256'] != sha(protocol_path):
        raise ValueError('activation input binding differs')
    if ledger['source_sha256'] != approval['source_sha256'] or ledger['cost_sha256'] != approval['cost_sha256']:
        raise ValueError('activation source/cost differs')
    if ledger['deadline_epoch'] != ledger['start_epoch'] + BUDGET_SECONDS or ledger['budget_seconds'] != BUDGET_SECONDS:
        raise ValueError('activation budget differs')
    if ledger['output'] != str(pathlib.Path(approval['output']).resolve()):
        raise ValueError('activation output differs')
    if output is not None and ledger['output'] != str(pathlib.Path(output).resolve()):
        raise ValueError('output differs')
    if time.time() >= ledger['deadline_epoch']:
        raise TimeoutError('formal deadline reached; no reset or restart')


def initialization_for_task(initializer, dataset, arm, seed, basis):
    """Small testable routing adapter; scientific initializer is unchanged."""
    task_id(dataset, seed, arm)
    if arm == 'E' and basis is None:
        raise ValueError('E requires the approved frozen dictionary')
    return initializer(dataset, arm, seed, basis if arm == 'E' else None)


def verify_ticket(ticket, args, parent_pid, device, now):
    label = task_id(args.dataset, args.train_seed, args.arm)
    expected = dict(parent_pid=parent_pid, dataset=args.dataset, train_seed=args.train_seed,
                    arm=args.arm, task_id=label, device=device,
                    output=str(pathlib.Path(args.output).resolve()),
                    approval_sha256=sha(args.approval), cost_sha256=sha(args.cost))
    if any(ticket.get(k) != v for k, v in expected.items()) or not 0 <= now-ticket['admitted_epoch'] <= 60:
        raise PermissionError('valid current launcher admission required')
