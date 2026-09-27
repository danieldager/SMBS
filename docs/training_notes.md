# Training notes

## LSTM: the loss plateau (February 2026)

Early LSTM runs on SpidR units stalled. Over 30+ short runs, the configuration from the
literature (embedding 200, hidden 1024, 3 layers, learning rate 1e-4, weight decay 0.01,
β2 0.98, no gradient clipping, inverse-square-root schedule) sat on a plateau near a
loss of 5.1. The configuration below converged immediately.

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Model size** | 256/256/2 (emb/hidden/layers) | Smaller converged better at this budget |
| **Optimiser** | Adam | AdamW runs in the same period plateaued at ~5.1 loss |
| **Learning rate** | 1e-2 | 10-100x higher than typical |
| **β2** | 0.99 | Slightly lower than the default 0.999 |
| **Gradient clip** | 5.0 | Needed for stability at this learning rate |
| **Dropout** | 0.0 | |
| **Weight decay** | 0.0 | |
| **LR schedule** | Constant, no warmup | |
| **Batch size** | 32×1×3 = 96 effective | per-device × accumulation × GPUs |

Loss with this recipe: about 1.9 at step 40, 1.7 at step 100, 1.65 at step 1000,
and no plateau across 5 random seeds.

Other observations from those runs:

1. Smaller models converged better: 256/256/2 over the published 200/1024/3.
2. A high learning rate was essential: 1e-2 against the published 1e-4.
3. No warmup was needed.
4. Orthogonal initialisation helped, but less than the optimiser settings.
5. Forget-gate bias initialised to 1.0 (standard LSTM practice).

**Caveat.** With weight decay at 0, PyTorch's `Adam` and `AdamW` compute the same
update, so the optimiser class alone cannot explain the plateau; the recipe changed
the learning rate, β2, clipping, weight decay and model size together. `smbs grid`
separates these: it runs the published configuration, the recipe above, and one-factor
ablations of the recipe (low learning rate, no clipping, with weight decay, β2 0.98,
and model size crossed with training recipe) over three seeds each. The packaged grid
uses `adamw_torch` for every configuration.

**Current defaults.** `smbs train --arch lstm` uses the published configuration
(hidden 1024, 3 layers, AdamW at 1e-4, up to 100,000 steps with early stopping). The
March LSTM results in the README are hidden-1024 models, i.e. this configuration's size,
not the small recipe above.

## GPT-2 configuration

| Parameter | Value |
|-----------|-------|
| **Model** | 768/12/12 (emb/layers/heads) |
| **Optimiser** | AdamW |
| **Learning rate** | 1e-4 |
| **β2** | 0.98 |
| **Gradient clip** | 0.0 (disabled) |
| **Weight decay** | 0.01 |
| **LR schedule** | inverse_sqrt with 1000 warmup steps |
| **Batch size** | 32 per device × 4 accumulation (16 × 8 for vocabularies above 1,000 units) |
| **Max steps** | 100,000 |

Checkpoints are saved every 1,000 steps and training stops early after 3 evaluations
without improvement. BF16 is on.
