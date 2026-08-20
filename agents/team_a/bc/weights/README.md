# Weights

`*.npz` is git-ignored by repository policy (model weights never in git), so this folder is
empty on a fresh clone and `main.py` will not run until you train a member:

```bash
uv run python agents/team_a/bc/extract.py --teacher tetsuya
uv run python agents/team_a/bc/train.py \
    --rows "$GUILDHALL_WAREHOUSE/datasets/kaggriculture/bc-rows/tetsuya-fold0.npz" \
    --out agents/team_a/bc/weights/tetsuya-a.npz
```

Training is deterministic (`--seed`, default 0) and takes ~30 s on CPU, so the weights are
reproducible from the replay corpus rather than stored. Every `.npz` in this folder becomes
an ensemble member; `BC_MEMBERS` (os.pathsep-separated paths) overrides the selection.

## Which corpus — the recipe that wins (engine 1.32.7)

The training **corpus** is the largest lever measured on this line; the architecture was
unchanged across all three rows below. Confirm = 400 episodes, held-out seeds 1000–1199,
paired with side swap, vs `TeamA-Farmer-v0` (exp-001).

| corpus | exp | val op top1 | confirm vs v0 |
|---|---|---|---|
| `teamA3` — 62 eps, 3 LB-top teachers pooled | exp-006b | 0.894 | **96.5% [94.2–97.9]**, bank 69,302 vs 36,131 |
| `broad` — 80 eps, 30 submissions | exp-006a | 0.982 | 98.5% [96.8–99.3], bank 60,636 vs 37,232 |
| `tetsuya` — 30 eps, one teacher | exp-003b | 0.906 | 40.5% [35.8–45.4] — loses |

`teamA3` is the one to train: it beats `broad` **95.2% [92.7–96.9]** head-to-head despite
scoring *worse* on both val accuracy and win rate against v0. Two traps that implies:

- **Offline accuracy is a filter, not a ranking.** Here it is anti-correlated with play.
- **Win rate against a common third opponent cannot separate two agents that both saturate
  against it** (98.5% vs 96.5% ranked these two backwards). Once a candidate is far past the
  incumbent, rank by a direct match. The bank against the shared opponent was the better hint.

Weights and their md5s are archived at `$GUILDHALL_WAREHOUSE/models/team-a-bc/`.
Full rows: `workshop/notes/experiments.md` (exp-006a, exp-006b).

**Open question for packaging:** a submission tarball must contain the weights (~0.74 MB per
member). Either the `*.npz` policy gets an exception for `agents/*/weights/`, or
`scripts/make_submission.py` pulls them from the warehouse at package time.
