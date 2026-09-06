"""Build a nightly reachability filter; no games or promotion by default."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from benchmark_brain_dispatch_cost import load
from compare_paired import compare, read

ROOT = Path(__file__).resolve().parents[1]


def build(source):
    replacements = [
        ('def _assign(positions, inventories, tasks):',
         'def _assign(positions, inventories, tasks, hours_left=23):', 1),
        ('LaborB1._assign(positions, inventories, tasks)',
         'LaborB1._assign(positions, inventories, tasks, 23 - view.hour)', 2),
        ('                    cluster_cost = 0 if idx == owner else 1',
         '                    if distance > hours_left:\n'
         '                        continue\n'
         '                    cluster_cost = 0 if idx == owner else 1', 1),
        ('                          for p in targets if p != positions[idx]]',
         '                          for p in targets if p != positions[idx]\n'
         '                          and abs(positions[idx][0] - p[0]) + abs(positions[idx][1] - p[1]) <= hours_left]', 1),
    ]
    for old, new, count in replacements:
        assert source.count(old) == count, old
        source = source.replace(old, new)
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', default='bots/brain/main.py')
    parser.add_argument('--run', action='store_true', help='Run only after other experiments finish')
    args = parser.parse_args()
    source = build((ROOT/args.baseline).read_text(encoding='utf-8-sig'))
    path = ROOT/'bots/brain_reachable_work/main.py'
    compile(source, str(path), 'exec')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding='utf-8')
    bot = load(path, 'reachable_work')
    task = (0, (1, 0), ('WATER',), None, None)
    assert bot.LaborB1._assign(((0, 0),), ({},), [task], 0) == {}
    assert bot.LaborB1._assign(((0, 0),), ({},), [task], 1) == {0: ['EAST']}
    assert bot.LaborB1._assign(((1, 0),), ({},), [task], 0) == {0: ['WATER']}
    assert bot.LaborB1._assign(((0, 0),), ({},), [], 0) == {}
    replay = json.loads((ROOT/'data/results/brain-own-diagnostic/main_vs_main_seed51.json').read_text())
    observations = [step[0]['observation'] for step in replay['steps'][:-1]]
    for obs in observations:
        action = bot._policy(obs, time.perf_counter())
        assert isinstance(action, dict) and 'market' in action
    out = ROOT/'data/results/brain-reachable-work'
    out.mkdir(parents=True, exist_ok=True)
    report = dict(policy_calls=len(observations), boundary_checks=4,
                  decision='Prepared only; paired game measurement required.')
    (out/'offline-checks.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))
    if args.run:
        arena = ROOT/'data/arena-reachable-work'
        environment = dict(os.environ, KAGGRI_ARENA_DIR=str(arena))
        tags = {name: f'reachable-work-{name}-screen-16' for name in ('control', 'candidate')}
        assert not any((ROOT/'data/results'/tag/'results.json').exists() for tag in tags.values()), 'Existing screen results; do not overwrite'
        control = ROOT/'bots/brain_reachable_control/main.py'
        control.parent.mkdir(parents=True, exist_ok=True)
        control.write_text((ROOT/args.baseline).read_text(encoding='utf-8-sig'), encoding='utf-8')
        for name, agent in [('control', control), ('candidate', path)]:
            with (out/f'{name}.log').open('w', encoding='utf-8') as log:
                subprocess.run([sys.executable, '-X', 'utf8', str(ROOT/'scripts/evaluate.py'),
                                '--agent', str(agent), '--opponents', 'bots/v46/main.py',
                                'bots/tapeselect/main.py', '--n', '16', '--seed-base', '210',
                                '--workers', '2', '--tag', tags[name]], cwd=ROOT, env=environment,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            result = json.loads((ROOT/'data/results'/tags[name]/'results.json').read_text())
            print(name, result['overall'], flush=True)
        candidate_games, candidate_stats = read(arena, tags['candidate'])
        control_games, control_stats = read(arena, tags['control'])
        paired = dict(per_opponent=compare(candidate_games, control_games),
                      candidate=candidate_stats, control=control_stats,
                      decision='Screen only; no automatic promotion or submission.')
        (out/'paired.json').write_text(json.dumps(paired, indent=2), encoding='utf-8')
        print(json.dumps(paired), flush=True)


if __name__ == '__main__':
    main()
