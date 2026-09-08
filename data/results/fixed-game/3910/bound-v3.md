# Bound v3

## Bounds

| seed | best play | v3 upper bound | gap | gap % | variables | constraints |
|---:|---:|---:|---:|---:|---:|---:|
| 3910 | $139,281 | $377,805.21 | $238,524.21 | 171.25% | 61,287 | 97,558 |
| 5000 | $126,610 | $398,877.26 | $272,267.26 | 215.04% | 61,287 | 101,494 |
| 5001 | $136,502 | $383,988.63 | $247,486.63 | 181.31% | 61,287 | 98,671 |
| 5002 | $95,681 | $381,468.02 | $285,787.02 | 298.69% | 61,287 | 103,592 |

## Product bounds: units / average price

| seed | wheat | carrot | tomato | strawberry | melon | egg | milk | wool | fertilizer |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3910 | 837.191 / $25.83 | 443.000 / $53.75 | 0 / $0 | 57.000 / $170.42 | 155.000 / $212.95 | 2,912.689 / $38.71 | 273.000 / $225.70 | 653.000 / $240.21 | 493.000 / $50.80 |
| 5000 | 817.313 / $30.74 | 166.443 / $47.23 | 0 / $0 | 400.000 / $266.74 | 139.000 / $224.33 | 1,662.256 / $42.56 | 378.000 / $249.84 | 327.000 / $229.37 | 898.474 / $28.33 |
| 5001 | 773.861 / $29.57 | 203.009 / $46.28 | 0 / $0 | 311.509 / $240.68 | 144.000 / $220.94 | 1,864.518 / $43.23 | 129.000 / $178.79 | 651.000 / $240.70 | 1,057.504 / $24.22 |
| 5002 | 746.280 / $37.71 | 264.188 / $56.50 | 114.000 / $280.43 | 378.667 / $304.59 | 128.000 / $231.24 | 1,579.913 / $41.30 | 324.000 / $239.34 | 75.000 / $186.24 | 674.788 / $37.38 |

## Binding constraints

| seed | zero-cash days | labour days | tile days | pool products |
|---:|---|---|---|---|
| 3910 | 0-9, 11 | 0, 2-29 | 0-2, 4-29 | 9/9 |
| 5000 | 0-11 | 0, 2-29 | 0-2, 5, 7-29 | 9/9 |
| 5001 | 0-10 | 0, 2-29 | 0-2, 5, 7-29 | 9/9 |
| 5002 | 0-11 | 0, 2-29 | 0-2, 5-28 | 9/9 |

## Validation

| seed | executed bank | replayed bank | minimum end-day bank | model-feasible | bank <= bound | violations |
|---:|---:|---:|---:|---|---|---:|
| 3910 | $139,281 | $139,281 | $181 | yes | yes | 0 |
| 5000 | $126,610 | $126,610 | $128 | yes | yes | 0 |
| 5001 | $136,502 | $136,502 | $161 | yes | yes | 0 |

Constraint-family checks: cash 3/3; animal ramp/housing 3/3; feed 3/3; labour 3/3; geometric travel 3/3; pool pricing 3/3; order slots 3/3.

## 3910 residual slack estimate

| relaxation | dollars | next proof constraint |
|---|---:|---|
| pooled tile-service routes | $95,000 | tile/day service flow with shared walks and shed-start paths |
| pooled animal identity, feed, care, held stock | $60,000 | animal/tile/day state network with feed-care-production transitions |
| crop cap timing, watering and fertilizer pooling | $31,000 | crop/tile/day growth-state network |
| fractional assets, same-day netting and omitted shed capacity | $20,000 | integer assets plus within-day inventory/cash checkpoints |
| drain-before-sale price envelope and $17 wheat-buy floor | $10,324 | turn-event pool inventory with buy/sell spread and 100-unit shed state |
| wheat carry safety ceiling | $22,200 | integrate buy/sell inventory and cash instead of adding the per-drain-event ceiling |
| total | $238,524 | |

Target miss: 3910 is 171.25% above play, not within 25%. Farm production is $355,605.21 and the separate spread/slot/shed-safe wheat carry ceiling is $22,200. The residual witness earns $112,736.82 from 2,912.689 eggs while simultaneously saturating the milk and wool drain envelopes; the current `max(0, distinct tiles - workers)` travel cut does not couple feed, care, harvest, crop work, shed visits, or worker endpoints on the same tile route. The next tightening must be a tile/day service-flow relaxation, not another scalar travel multiplier.

No MIP run: the LP misses the target by $238,524.21; integrality alone cannot remove the identified route/state pooling slack.
