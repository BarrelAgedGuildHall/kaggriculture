"""Original labor-allocation experiments against an immutable Brain v2.

Run four policies on the same 10 tuning seeds in both seats. Sequential arena
registration avoids shared JSON races. The forecast uses no opponent code/traces.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/results/brain-dispatch-screen'

VALUE_ASSIGN = '''    @staticmethod
    def _assign(positions, inventories, tasks, view=None):
        """Assign feasible jobs by current reward per travel-and-work action."""
        tiles = _labor_dict(view.tiles)
        prices = dict(view.market_prices)
        day, hour = view.day, view.hour
        values = []
        emergencies = []
        for priority, target, action, need, skip in tasks:
            x, y = target
            tile = tiles[y][x]
            tile = tile if isinstance(tile, dict) else {}
            op = action[0]
            value = 20.0
            emergency = 0
            if op == 'HARVEST':
                product = tile.get('crop')
                if tile.get('animal') in ANIMALS:
                    product = ANIMALS[tile['animal']]['product']
                value = prices.get(product, 30) * max(1, tile.get('yield_units', 0))
            elif op == 'WATER':
                crop = tile.get('crop')
                spec = CROPS.get(crop, CROPS['WHEAT'])
                age = day - tile.get('planted_day', day)
                grows = (not spec['ongoing'] and WINDOW_START[crop] <= age <= TARGET_AGE[crop])
                value = prices.get(crop, 30) if grows else 12.0
                # Only an actual survival deadline gets unconditional precedence.
                emergency = int(tile.get('consecutive_unwatered', 0) >= 1 and hour >= 12)
            elif op in ('FEED', 'CARE'):
                animal = ANIMALS.get(tile.get('animal'), ANIMALS['COW'])
                value = prices.get(animal['product'], 40) / animal['interval']
                if op == 'FEED':
                    value += prices.get('FERTILIZER', 20) * 0.5
                    emergency = int(tile.get('consecutive_unfed', 0) >= 1 and hour >= 12)
            elif op == 'COLLECT_FERTILIZER':
                value = prices.get('FERTILIZER', 30)
            elif op == 'FERTILIZE':
                value = max(30, prices.get('FERTILIZER', 30))
            elif op == 'PICKUP':
                value = 50.0 if action[1] == 'WHEAT' else 35.0
            elif op == 'PLANT':
                crop = action[1]
                spec = CROPS[crop]
                value = max(8, (prices.get(crop, 30) * spec['max_yield'] - spec['seed'])
                            / (TARGET_AGE[crop] + 2))
            elif op in ('PLACE', 'BUILD_PASTURE'):
                value = 35.0
            values.append(value)
            emergencies.append(emergency)
        available = set(range(len(positions)))
        pending = set(range(len(tasks)))
        actions = {}
        while available and pending:
            best = None
            for ti in pending:
                _priority, target, action, need, skip = tasks[ti]
                for idx in available:
                    if need is not None and inventories[idx].get(need, 0) <= 0:
                        continue
                    if skip is not None and inventories[idx].get(skip, 0) > 0:
                        continue
                    distance = abs(positions[idx][0]-target[0]) + abs(positions[idx][1]-target[1])
                    if distance > 23 - hour:
                        continue
                    score = (emergencies[ti], values[ti] / (distance + 1), -distance, -idx, -ti)
                    if best is None or score > best[0]:
                        best = (score, idx, ti, distance)
            if best is None:
                break
            _score, idx, ti, distance = best
            action = list(tasks[ti][2]) if distance == 0 else LaborB1._move(positions[idx], tasks[ti][1])
            actions[idx] = action
            available.remove(idx)
            pending.remove(ti)
        return actions

'''


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = (ROOT / 'bots/brain_fertilizer/main.py').read_text(encoding='utf-8')
    old = 'choices.append((priority, cluster_cost, distance, idx, ti))'
    assert source.count(old) == 1
    variants = {
        'control': source,
        'nearest': source.replace(old, 'choices.append((priority, distance, cluster_cost, idx, ti))'),
        'soft': source.replace(old, 'choices.append((priority * 2 + distance + cluster_cost, priority, distance, idx, ti))'),
    }
    begin = source.index('    @staticmethod\n    def _assign(')
    end = source.index('    @staticmethod\n    def dispatch(', begin)
    value_source = source[:begin] + VALUE_ASSIGN + source[end:]
    value_source = value_source.replace('LaborB1._assign(positions, inventories, tasks)',
                                        'LaborB1._assign(positions, inventories, tasks, view)')
    variants['value'] = value_source
    env = dict(os.environ)
    env['KAGGRI_ARENA_DIR'] = str(ROOT / 'data/arena-dispatch-screen')
    rows = []
    for name, text in variants.items():
        path = ROOT / 'bots' / f'brain_dispatch_{name}' / 'main.py'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')
        compile(text, str(path), 'exec')
        tag = f'dispatch-{name}-screen-20'
        with (OUT / f'{name}.log').open('w', encoding='utf-8') as log:
            run = subprocess.run([sys.executable, str(ROOT/'scripts/evaluate.py'), '--agent', str(path),
                                  '--opponents', 'bots/v46/main.py', '--n', '20', '--seed-base', '50',
                                  '--workers', '10', '--tag', tag], cwd=ROOT, env=env,
                                 stdout=log, stderr=subprocess.STDOUT)
        result_path = ROOT/'data/results'/tag/'results.json'
        if not result_path.exists():
            raise RuntimeError(f'No game measurement for {name}; inspect log')
        summary = json.loads(result_path.read_text())
        row = dict(name=name, path=str(path), run_id=summary['run_id'], returncode=run.returncode,
                   **summary['per_opponent'][0])
        row['margin'] = row['money_us_mean'] - row['money_them_mean']
        rows.append(row)
        (OUT/'results.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
        print(f"{name}: wins={row['win_rate']:.1%} margin={row['margin']:.0f} "
              f"crash={row['crashes']} timeout={row['timeouts']}", flush=True)


if __name__ == '__main__':
    main()
