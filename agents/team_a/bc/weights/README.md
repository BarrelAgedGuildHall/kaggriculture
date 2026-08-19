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

**Open question for packaging:** a submission tarball must contain the weights (~0.74 MB per
member). Either the `*.npz` policy gets an exception for `agents/*/weights/`, or
`scripts/make_submission.py` pulls them from the warehouse at package time.
