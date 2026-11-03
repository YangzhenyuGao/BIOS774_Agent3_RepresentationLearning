import importlib.util
from types import SimpleNamespace

from agent3_replearn.utils import ROOT


def script_module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_serial_array_limits_concurrency_without_discounting_task_hours(monkeypatch):
    module = script_module('gpu_jobs')
    jobs = [dict(job_id='1', state='PENDING', gpus=1, wall_minutes=360, gpu_hours=0),
            *[dict(job_id=f'2_{i}', array_parent='2', array_parallel=1, state='PENDING',
                   gpus=1, wall_minutes=120, gpu_hours=0) for i in range(2)]]
    monkeypatch.setattr(module.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(stdout=''))
    result = module.reconcile(dict(jobs=jobs, gpu_hours_cap=48))
    assert result['gpu_slots_reserved'] == 2
    assert result['gpu_hours_committed'] == 10
    for job in jobs[1:]:
        job['array_parallel'] = 2
    assert module.reserved_gpu_slots(jobs) == 3


def test_cancelled_unstarted_array_is_expanded_from_parent_and_releases_reservation(monkeypatch):
    module = script_module('gpu_jobs')
    calls = []

    def accounting(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout='2_0|CANCELLED by 1|0||h100_priority\n2_1|CANCELLED by 1|0||h100_priority\n')

    monkeypatch.setattr(module.subprocess, 'run', accounting)
    jobs = [dict(job_id=f'2_{i}', array_parent='2', array_parallel=1, state='PENDING',
                 gpus=1, wall_minutes=240, gpu_hours=0, partition='h100_priority') for i in range(2)]
    result = module.reconcile(dict(jobs=jobs, gpu_hours_cap=48))
    assert result['gpu_slots_reserved'] == result['gpu_hours_committed'] == result['gpu_hours_used'] == 0
    assert all('--array' in command and command[command.index('-j') + 1] == '2' for command in calls)


def test_measured_cap_fits_queue_with_safety_and_startup_allowance():
    module = script_module('adjust_wsi_resources')
    assert module.choose_cap({'64': 1.97, '128': 3.94, '256': 7.88}) == 128
    assert module.choose_cap({'256': 1.0}) == 256
    try:
        module.choose_cap({'64': 10.})
    except RuntimeError:
        pass
    else:
        raise AssertionError('An infeasible measured cap was silently accepted')


def test_throttle_increase_is_rejected_until_pbmc_releases_its_gpu(monkeypatch):
    module = script_module('gpu_jobs')
    commands = []
    monkeypatch.setattr(module.subprocess, 'run', lambda command, **kwargs: commands.append(command))
    jobs = [dict(job_id='1', state='RUNNING', gpus=1),
            *[dict(job_id=f'2_{i}', array_parent='2', array_parallel=1, state='PENDING', gpus=1) for i in range(2)]]
    ledger = dict(jobs=jobs, concurrent_gpu_cap=2)
    try:
        module.set_array_parallel(ledger, '2', 2)
    except RuntimeError:
        pass
    else:
        raise AssertionError('Throttle changed while PBMC plus two feature tasks would require three GPUs')
    assert not commands
    jobs[0]['state'] = 'COMPLETED'
    module.set_array_parallel(ledger, '2', 2)
    assert commands == [['scontrol', 'update', 'JobId=2', 'ArrayTaskThrottle=2']]
    assert module.reserved_gpu_slots(jobs) == 2
