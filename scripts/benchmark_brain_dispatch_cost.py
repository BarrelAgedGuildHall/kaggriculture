"""Check and time removal of discarded legacy dispatch work on saved states.

This is behavior-equivalence verification, not strategy training or a game win
measurement. Neither the retained bot nor running arena snapshots are modified.
"""
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    baseline_path = ROOT/'bots/brain_late_labor_keep12/main.py'
    text = baseline_path.read_text(encoding='utf-8')
    old = 'while idx < total and free and not budget_hit:'
    new = ('while idx < total and free and not budget_hit and not '
           '(labor_module is LaborB1 and not (last_day and hour >= DROP_HOUR)):')
    assert text.count(old) == 1
    path = ROOT/'bots/brain_fast_dispatch/main.py'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.replace(old, new), encoding='utf-8')
    baseline = load(baseline_path, 'dispatch_cost_baseline')
    candidate = load(path, 'dispatch_cost_candidate')
    replay = json.loads((ROOT/'data/results/brain-own-diagnostic/main_vs_main_seed51.json').read_text())
    states = [step[seat]['observation'] for step in replay['steps'][:-1] for seat in (0, 1)]
    mismatches = []
    for index, obs in enumerate(states):
        expected = baseline._policy(obs, time.perf_counter())
        actual = candidate._policy(obs, time.perf_counter())
        if actual != expected:
            mismatches.append(index)
    samples = {'baseline': [], 'candidate': []}
    # Interleave process-CPU measurements so other running processes do not
    # inflate the estimate and order/cache bias is reduced.
    for repeat in range(6):
        order = [('baseline', baseline), ('candidate', candidate)]
        if repeat % 2:
            order.reverse()
        for name, module in order:
            started = time.process_time()
            for obs in states:
                module._policy(obs, time.perf_counter())
            samples[name].append(time.process_time()-started)
    medians = {name: statistics.median(values) for name, values in samples.items()}
    report = dict(observations=len(states), mismatches=mismatches, cpu_seconds=samples,
                  median_cpu_seconds=medians,
                  relative_cpu_reduction=1-medians['candidate']/medians['baseline'],
                  decision='Offline check only; standalone game verification required before integration.')
    out = ROOT/'data/results/brain-dispatch-cost'
    out.mkdir(parents=True, exist_ok=True)
    (out/'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    assert not mismatches, 'Actions differ; do not integrate'


if __name__ == '__main__':
    main()
