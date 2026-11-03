"""Submit/reconcile project GPU jobs with an aggregate reservation and runtime cap."""
import argparse
import fcntl
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / 'logs/gpu_ledger.json'
ACTIVE = {'PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED', 'REQUEUED', 'RESIZING'}


def reserved_gpu_slots(jobs):
    """Array throttle bounds concurrency; all tasks still count toward GPU-hours."""
    total, arrays = 0, {}
    for job in jobs:
        if job['state'] not in ACTIVE:
            continue
        parent = job.get('array_parent')
        if not parent:
            total += job['gpus']
            continue
        group = arrays.setdefault(parent, dict(requested=0, ceiling=0))
        group['requested'] += job['gpus']
        group['ceiling'] = max(group['ceiling'], job['gpus'] * job.get('array_parallel', 2))
    return total + sum(min(group['requested'], group['ceiling']) for group in arrays.values())


def set_array_parallel(ledger, parent, parallel):
    matched = [job for job in ledger['jobs'] if str(job.get('array_parent')) == str(parent)]
    if not matched:
        raise ValueError('Array is not enrolled in the project ledger')
    if not any(job['state'] in ACTIVE for job in matched):
        return
    planned = [dict(job) for job in ledger['jobs']]
    for job in planned:
        if str(job.get('array_parent')) == str(parent):
            job['array_parallel'] = parallel
    if reserved_gpu_slots(planned) > ledger['concurrent_gpu_cap']:
        raise RuntimeError('Array throttle would exceed the project GPU concurrency cap')
    subprocess.run(['scontrol', 'update', f'JobId={parent}', f'ArrayTaskThrottle={parallel}'],
                   capture_output=True, text=True, check=True)
    for job in matched:
        if job.get('array_parallel', 2) != parallel:
            job.setdefault('parallelism_changes', []).append(dict(previous=job.get('array_parallel', 2),
                current=parallel, time_utc=datetime.now(timezone.utc).isoformat()))
        job['array_parallel'] = parallel


def reconcile(ledger):
    for job in ledger['jobs']:
        query_id = job.get('array_parent') or job['job_id']
        run = subprocess.run(['sacct', '--array', '-X', '-j', str(query_id), '-n', '-P',
                              '--format=JobID,State,ElapsedRaw,AllocTRES,Partition'],
                             capture_output=True, text=True, check=True)
        for line in run.stdout.splitlines():
            fields = line.split('|')
            if len(fields) < 4 or fields[0] != str(job['job_id']):
                continue
            job['state'] = fields[1].split()[0].rstrip('+')
            job['elapsed_seconds'] = int(fields[2])
            job['alloc_tres'] = fields[3]
            if len(fields) > 4 and fields[4]:
                if job['partition'] != fields[4]:
                    job.setdefault('partition_changes', []).append(dict(previous=job['partition'], current=fields[4]))
                job['partition'] = fields[4]
            job['gpu_hours'] = job['gpus'] * job['elapsed_seconds'] / 3600
    ledger['gpu_hours_used'] = sum(j.get('gpu_hours', 0) for j in ledger['jobs'])
    ledger['gpu_hours_committed'] = sum(
        j['gpus'] * j['wall_minutes'] / 60 if j['state'] in ACTIVE else j.get('gpu_hours', 0)
        for j in ledger['jobs'])
    ledger['gpu_slots_reserved'] = reserved_gpu_slots(ledger['jobs'])
    ledger['reconciled_utc'] = datetime.now(timezone.utc).isoformat()
    if ledger['gpu_hours_used'] > ledger['gpu_hours_cap']:
        raise RuntimeError('GPU-hour cap exceeded; user approval required before further GPU work')
    return ledger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--submit', nargs='+', help='sbatch script followed by its positional arguments')
    parser.add_argument('--purpose')
    parser.add_argument('--minutes', type=int)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--partition', default='h100_short',
                        choices=['h100_short', 'h100_priority', 'h100_preempt'])
    parser.add_argument('--cpus', type=int, default=8)
    parser.add_argument('--memory', default='64G')
    parser.add_argument('--dependency')
    parser.add_argument('--array-tasks', type=int, default=1)
    parser.add_argument('--array-parallel', type=int, choices=[1, 2], default=2)
    parser.add_argument('--set-array-parallel', type=int, choices=[1, 2])
    parser.add_argument('--array-parent')
    args = parser.parse_args()
    LEDGER.parent.mkdir(exist_ok=True)
    with (LEDGER.parent / 'gpu_ledger.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = json.loads(LEDGER.read_text()) if LEDGER.exists() else dict(
            gpu_hours_cap=48, concurrent_gpu_cap=2, jobs=[])
        reconcile(ledger)
        if args.set_array_parallel is not None:
            if args.submit or not args.array_parent:
                raise ValueError('Throttle changes require an enrolled array parent and no new submission')
            set_array_parallel(ledger, args.array_parent, args.set_array_parallel)
            reconcile(ledger)
            print('Array throttle updated or already complete:', args.array_parent)
        elif args.submit:
            if not args.minutes or not args.purpose or args.gpus not in {1, 2}:
                raise ValueError('Submission requires purpose, positive wall minutes, and one or two GPUs')
            if args.array_tasks not in {1, 2}:
                raise ValueError('At most two GPU array tasks may run concurrently')
            allocation = args.gpus * args.array_tasks
            concurrent_allocation = args.gpus * min(args.array_tasks, args.array_parallel)
            if reserved_gpu_slots(ledger['jobs']) + concurrent_allocation > ledger['concurrent_gpu_cap']:
                raise RuntimeError('At most two reserved/running GPUs; wait for existing jobs')
            if ledger['gpu_hours_committed'] + allocation * args.minutes / 60 > ledger['gpu_hours_cap']:
                raise RuntimeError('Reservation would exceed 48 GPU-hours; ask user before exceeding')
            command = ['sbatch', '--parsable', '--account=h100_harlin', f'--partition={args.partition}',
                       f'--gres=gpu:{args.gpus}', f'--cpus-per-task={args.cpus}', f'--mem={args.memory}',
                       f'--time={args.minutes}']
            if args.dependency:
                command.append(f'--dependency=afterok:{args.dependency}')
            if args.array_tasks > 1:
                command.append(f'--array=0-{args.array_tasks - 1}%{args.array_parallel}')
            command.extend(args.submit)
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
            jid = result.stdout.strip().split(';')[0]
            for task in range(args.array_tasks):
                ledger['jobs'].append(dict(job_id=jid if args.array_tasks == 1 else f'{jid}_{task}',
                                          array_parent=jid if args.array_tasks > 1 else None,
                                          array_parallel=min(args.array_tasks, args.array_parallel),
                                          partition=args.partition, gpus=args.gpus,
                                          wall_minutes=args.minutes, purpose=args.purpose,
                                          state='PENDING', gpu_hours=0, elapsed_seconds=0,
                                          command=command, submitted_utc=datetime.now(timezone.utc).isoformat()))
            reconcile(ledger)
            print(jid)
        else:
            print(json.dumps(ledger, indent=2))
        temporary = LEDGER.with_suffix('.tmp')
        temporary.write_text(json.dumps(ledger, indent=2))
        temporary.replace(LEDGER)


if __name__ == '__main__':
    main()
