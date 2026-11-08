# Training sampling and validation

All features are opt-in; existing checkpoints, model keys, shapes, and default
training behavior are unchanged.

* `--sync-fix` accumulates detached metric tensors on the training device, converting
  them to Python only at log time. No loss or gradient calculation changes.
* `--sampler-v2` counts grid, Doom columns/sprites/meta, cursor and state changes.
  Doom columns/sprites count at one quarter per changed component. Window weights
  are `1 + min(activity, 10)`: an idle window has at least 1/11 of the busiest
  window's weight **within its mode**. Mode masses default to their frame shares.
  `--mode-mix '{"doom":0.2,"mixed":0.8}'` overrides masses (normalized; omitted
  modes receive zero). Mode names come from `.npz.json` sidecars, falling back to
  filenames. Explicit mode weights must name modes present in eligible windows.
* `--extra-data DIR --extra-shards 40 --extra-frac 0.15` reserves a fixed fraction
  of each batch for another source (nearest integer batch slots). The sorted last
  shard is excluded for validation; only the first N remaining shards load. N is
  restricted to 1..40 to bound memory. Future cats target: configurable 15–20%.
* `--holdout-episodes 1` reserves each shard's final episode; episodes, not windows,
  define the split, so no training window overlaps held-out frames. Keep this flag
  identical when evaluating the same directory. At least one episode per shard
  must remain for training. `--max-steps-data` remains a whole-shard soft limit.

## Validation options

Use held-out episodes of the same shard set:

```sh
python -m dreaming.train --data data/local --holdout-episodes 1 --sampler-v2 --sync-fix
python -m dreaming.eval --checkpoint runs/local/ckpt.pt --data data/local \
  --holdout-episodes 1 --out results.json
```

For the A/B split, select the first sorted shard per mode not present in training
plus the last dense cats shard (neither directory is modified):

```sh
python -m dreaming.eval --checkpoint runs/local/ckpt.pt --data /vol/shards \
  --exclude-data /vol/shards_first_5k --cats-data /vol/shards_cats_dense_v1 \
  --out results.json
```

Alternatively, `--generate-heldout` generates two fresh 2,000-step episodes per app
with reserved base seeds 100,000,000–100,000,006 in a separate `--data` directory.
`dataset.make_shard` derives episode seeds as `base_seed * 10,000,000 + shard *
1000 + episode`. With one shard and two episodes per mode, generated evaluation
episode seeds are 1,000,000,000,000,000 + mode_index * 10,000,000 + {0,1}.
These are disjoint from the standard training generation bases (1000+ container
index; specials 2000+); reserve these base seeds exclusively for evaluation.
Never include the generated directory in training or extra sources. Evaluation
samples uniformly with seed 7341, masks tile logits to
`ATLAS.used`, and reports per-mode grid CE, all/changed accuracy, and copy-last
accuracy over 2,000 windows. Autonomous rollouts use 20 starts and 30/120-step
horizons, predicted observations with recorded external actions, and per-step
accuracy alongside copy-first-frame accuracy. Empty changed sets report zero
changed accuracy and `changed_tiles=0` rather than NaN. Reduce `--windows`,
`--starts`, and `--horizons` for CPU smoke checks.

Tests: `OMP_NUM_THREADS=1 python -m pytest dreaming/test_train_eval.py -q`.
