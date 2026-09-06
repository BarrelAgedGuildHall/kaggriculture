"""Count eligible bundled work at own v6 travel destinations."""
import inspect
import json
import sys
import time
from collections import Counter
from pathlib import Path
from benchmark_brain_dispatch_cost import load

ROOT = Path(__file__).resolve().parents[1]


def main():
    bot = load(ROOT/'bots/brain_value_v6/main.py', 'bundle_diagnostic')
    lines, start = inspect.getsourcelines(bot.LaborB1._assign)
    points = {start+i for i, line in enumerate(lines) if line.strip() == 'pending.remove(ti)'}
    counts = Counter()
    histogram = Counter()

    def trace(frame, event, arg):
        if frame.f_code is not bot.LaborB1._assign.__code__:
            return None
        if event == 'line' and frame.f_lineno in points:
            v = frame.f_locals
            idx, ti = v['idx'], v['ti']
            target = v['tasks'][ti][1]
            if target != v['positions'][idx]:
                inv = v['inventories'][idx]
                eligible = sum(v['tasks'][j][1] == target
                               and (v['tasks'][j][3] is None or inv.get(v['tasks'][j][3], 0) > 0)
                               and (v['tasks'][j][4] is None or inv.get(v['tasks'][j][4], 0) <= 0)
                               for j in v['pending'])
                histogram[eligible] += 1
                counts['travel_assignments'] += 1
        return trace

    replay = json.loads((ROOT/'data/results/brain-v6-diagnostic/seat0.json').read_text(encoding='utf-8'))
    for step in replay['steps'][:-1]:
        sys.settrace(trace)
        try:
            bot._policy(step[0]['observation'], time.perf_counter())
        finally:
            sys.settrace(None)
        counts['observations'] += 1
    report = dict(counts=counts, eligible_jobs_at_destination=histogram,
                  scope='Own v6 seed3000 seat0 observations; pending eligibility snapshots, not guaranteed future execution or profit.')
    (ROOT/'data/results/brain-v6-diagnostic/bundles.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
