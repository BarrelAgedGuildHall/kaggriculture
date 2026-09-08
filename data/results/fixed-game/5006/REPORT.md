# Fixed game seed 5006, round-3 generalization

Best bank: **$93,886**, mixed starting family, independently replayed through the real engine with zero faults.

The 20.08-minute run used four workers because the Engineer's concurrent 5000-5005 batch already occupied eight. It completed 2,124 evaluations (105.8/min). Best-so-far was $76,684 at 5 minutes, $88,590 at 10, and $93,886 at 20. Family bests: mixed $93,886; round-2 incumbent $87,657; tomato/carrot $83,600; goose/carrot/sheep $57,449.

`pin-verification.json` confirms the pass trajectory is unchanged at all 720 observations and different action sequences receive the same pinned shops.
