# Fixed-game solved dataset

Each `data/results/fixed-game/<seed>/solved.json` is a self-contained JSON document with the pinned instance identity, shop schedule, season drain, verified result metrics, best day-level schedule, 720-action tape, and 720 `training_records`. A record has `turn` (0–719), `observation` (the full seat-0 pre-action engine observation converted recursively to ordinary JSON objects, arrays, strings, numbers, booleans, and null), `action` (`farmer`, `hands`, and `market` in the engine action schema), `applied`, and `terminal`; turns 0–718 are applied transitions, while turn 719 contains the terminal observation and the tape's padded PASS with `applied: false`.

To reconstruct or audit a state, load the sibling `instance.json`, initialize `scripts.fixed_game.sim.Sim(instance)`, and step through the applied actions in order; the observation before each step must equal that record's serialized `observation`, and after 719 transitions the simulator must equal record 719 and be DONE. Consumers training a policy should filter on `applied: true`; `terminal` is provided so sequence models can retain the final state without treating its padding action as a label.

Top-level result fields are:

- `pinned_shop_schedule` and `season_drain`: the fixed exogenous market instance extracted from the seed's unmodified pass/pass run.
- `best_bank`, `bank_curve`, `product_revenue`, `sold_units`, and `hands_per_day`: measurements from an independent real-interpreter replay of the selected tape.
- `bound`: the valid LP upper bound when requested and completed; `bound_error` explains a timeout.
- `best_schedule` and `schedule_profile`: the searched day-level intent and a compact crop-block/herd/hiring description.
- `actions`: exactly 720 actions. The engine consumes the first 719; the last is padding aligned with the terminal record.
