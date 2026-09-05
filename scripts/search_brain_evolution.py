"""Optimize our own observation-driven Brain using game rewards, never imitation.

Four generations of 16 policies, eight paired tuning seeds, two independent
opponents. Unseen seed bands validate the finalists and then the selected policy.
No replay actions, opponent source, or identity fingerprints enter the search.
Runs sequential arena registrations with a bounded worker count and caches exact
parameter repeats. Output is measurement data; STATUS.md tracks campaign state.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/results/brain-evolution'
DOMAINS = {
    'opening_wheat': [0.35, 0.5, 0.65, 0.8],
    'crop_wheat': [0.3, 0.45, 0.55, 0.7],
    'crop_melon': [0.0, 0.1, 0.2],
    'animals': [10, 14, 18, 22],
    'cow_share': [0.3333, 0.3889, 0.5, 0.6667],
    'reserve': [80, 160, 240, 400],
    'animal_ramp': [1, 2, 3],
    'peak_hands': [10, 12, 14],
    'late_hands': [8, 10, 12, 14],
    'task_value': [18.0, 24.0, 36.0],
    'harvest_min': [1, 2, 4],
    'feed_buffer': [1, 4, 8],
    'wheat_floor': [1, 20, 35],
    'pasture_buffer': [2, 6, 1000],
    'care_priority': [1, 2, 3],
}
DEFAULT = dict(opening_wheat=0.8, crop_wheat=0.55, crop_melon=0.2,
               animals=18, cow_share=0.3889, reserve=240, animal_ramp=2,
               peak_hands=12, late_hands=8, task_value=24.0, harvest_min=2,
               feed_buffer=4, wheat_floor=35, pasture_buffer=1000, care_priority=3)


def render(source, params):
    if params == DEFAULT:
        return source
    text = source

    def change(old, new):
        nonlocal text
        assert text.count(old) == 1, old
        text = text.replace(old, new)

    n_cows = round(params['animals'] * params['cow_share'])
    crop = {'WHEAT': params['crop_wheat'],
            'STRAWBERRY': round(1 - params['crop_wheat'] - params['crop_melon'], 4)}
    if params['crop_melon']:
        crop['MELON'] = params['crop_melon']
    change('{"WHEAT": 0.80, "MELON": 0.20}',
           repr({'WHEAT': params['opening_wheat'], 'MELON': round(1-params['opening_wheat'], 4)}))
    change('CROP_SHARE = {"WHEAT": 0.55, "STRAWBERRY": 0.25, "MELON": 0.20}',
           'CROP_SHARE = ' + repr(crop))
    change('MAX_ANIMALS = 18', f"MAX_ANIMALS = {params['animals']}")
    change('ANIMAL_TARGETS = {"COW": 7, "SHEEP": 11}',
           'ANIMAL_TARGETS = ' + repr({'COW': n_cows, 'SHEEP': params['animals'] - n_cows}))
    change('ANIMALS_RAMP = (2, 2)', f"ANIMALS_RAMP = (2, {params['animal_ramp']})")
    change('MONEY_RESERVE = 240', f"MONEY_RESERVE = {params['reserve']}")
    change('HIRES_PER_DAY = 12', f"HIRES_PER_DAY = {params['peak_hands']}")
    change('PHASE_CAPS = ((0, 6), (3, 8), (6, 10), (10, 12), (25, 10), (28, 8))',
           'PHASE_CAPS = ' + repr(((0, 6), (3, 8), (6, 10), (10, params['peak_hands']),
                                  (25, max(10, params['late_hands'])), (28, params['late_hands']))))
    change('TASK_VALUE = 24.0', f"TASK_VALUE = {params['task_value']}")
    change('ANIMAL_HARVEST_MIN = 2', f"ANIMAL_HARVEST_MIN = {params['harvest_min']}")
    change('WHEAT_FEED_BUFFER = 4', f"WHEAT_FEED_BUFFER = {params['feed_buffer']}")
    change('"WHEAT": 35, "CARROT": 2', f"\"WHEAT\": {params['wheat_floor']}, \"CARROT\": 2")
    change('add(3, x, y, ["CARE"])', f"add({params['care_priority']}, x, y, [\"CARE\"])")
    if params['pasture_buffer'] < 1000:
        change('need_coops = target_coops - len(coop_pos)',
               'need_coops = min(target_coops - len(coop_pos), max(0, '
               'sum(int(shed.get(a, 0)) + carried.get(a, 0) for a in ANIMAL_TARGETS) '
               f"+ {params['pasture_buffer']} - len(empty_coops)))")
    return text


def objective(per_opponent):
    # Balanced share of final bank avoids rewarding one unusually rich board or
    # a specialist that merely gives up against the other reference opponent.
    margins = [(r['money_us_mean']-r['money_them_mean']) /
               max(1, r['money_us_mean']+r['money_them_mean']) for r in per_opponent]
    return 0.75 * sum(margins) / len(margins) + 0.25 * min(margins)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = (ROOT/'bots/brain_fertilizer/main.py').read_text(encoding='utf-8')
    rng = random.Random(20260905)
    environment = dict(os.environ)
    environment['KAGGRI_ARENA_DIR'] = str(ROOT/'data/arena-brain-evolution')
    cache = {}
    all_rows = []
    policy_count = 0

    def draw():
        return {key: rng.choice(values) for key, values in DOMAINS.items()}

    def evaluate(params, stage, n, seed_base):
        nonlocal policy_count
        key = json.dumps(params, sort_keys=True)
        cache_key = (key, n, seed_base)
        if cache_key in cache:
            return cache[cache_key]
        name = f'brain_search_{policy_count:04d}'
        policy_count += 1
        directory = ROOT/'bots'/name
        directory.mkdir(parents=True, exist_ok=True)
        text = render(source, params)
        compile(text, name, 'exec')
        (directory/'main.py').write_text(text, encoding='utf-8')
        (directory/'parameters.json').write_text(json.dumps(params, indent=2), encoding='utf-8')
        tag = f'{name}-{stage}'
        command = [sys.executable, str(ROOT/'scripts/evaluate.py'), '--agent', str(directory/'main.py'),
                   '--opponents', 'bots/v46/main.py', 'bots/tapeselect/main.py',
                   '--n', str(n), '--seed-base', str(seed_base), '--workers', '12', '--tag', tag]
        with (OUT/f'{tag}.log').open('w', encoding='utf-8') as log:
            result = subprocess.run(command, cwd=ROOT, env=environment,
                                    stdout=log, stderr=subprocess.STDOUT)
        measurement = ROOT/'data/results'/tag/'results.json'
        if not measurement.exists():
            raise RuntimeError(f'No measurement for {tag}; inspect its log')
        data = json.loads(measurement.read_text())
        faults = sum(r['crashes'] + r['timeouts'] for r in data['per_opponent'])
        row = dict(name=name, stage=stage, path=str(directory/'main.py'), params=params,
                   n=n, seed_base=seed_base, run_id=data['run_id'], per_opponent=data['per_opponent'],
                   objective=objective(data['per_opponent']) if not faults else -10,
                   faults=faults, returncode=result.returncode)
        cache[cache_key] = row
        all_rows.append(row)
        (OUT/'measurements.json').write_text(json.dumps(all_rows, indent=2), encoding='utf-8')
        print(f"{name} {stage}: objective={row['objective']:.4f}, "
              f"wins={[r['win_rate'] for r in row['per_opponent']]}, faults={faults}", flush=True)
        return row

    population = [DEFAULT.copy()] + [draw() for _ in range(15)]
    generations = []
    for generation in range(4):
        rows = [evaluate(p, f'g{generation}', 16, 150) for p in population]
        ranked = sorted(rows, key=lambda r: r['objective'], reverse=True)
        generations.append(ranked)
        (OUT/'generations.json').write_text(json.dumps(generations, indent=2), encoding='utf-8')
        print(f"GENERATION {generation} best={ranked[0]['name']} "
              f"objective={ranked[0]['objective']:.4f}", flush=True)
        elites = [r['params'] for r in ranked[:4]]
        population = [p.copy() for p in elites] + [draw() for _ in range(4)]
        while len(population) < 16:
            left, right = rng.sample(elites, 2)
            child = {key: rng.choice((left[key], right[key])) for key in DOMAINS}
            for key in rng.sample(list(DOMAINS), rng.randint(2, 5)):
                child[key] = rng.choice(DOMAINS[key])
            if child not in population:
                population.append(child)

    training = sorted((row for row in all_rows if row['seed_base'] == 150),
                      key=lambda row: row['objective'], reverse=True)
    finalists = []
    for row in training:
        if row['params'] not in finalists:
            finalists.append(row['params'])
        if len(finalists) >= 4:
            break
    validation_params = [DEFAULT.copy()] + [p for p in finalists if p != DEFAULT]
    validation = [evaluate(p, 'validation', 40, 1500) for p in validation_params]
    winner = max(validation, key=lambda row: row['objective'])
    final_params = [DEFAULT.copy()] + ([winner['params']] if winner['params'] != DEFAULT else [])
    final = [evaluate(p, 'final', 100, 1600) for p in final_params]
    summary = dict(training_best=training[0], validation=validation, final=final,
                   decision='Final paired results require review; no automatic integration or submission.')
    (OUT/'final.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('SEARCH COMPLETE: review final.json and paired seed deltas before promotion.', flush=True)


if __name__ == '__main__':
    main()
