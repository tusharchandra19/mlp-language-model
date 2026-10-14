# Main: 5-character MLP with momentum

**Notebook:** [`3_layer_nn_character_pred.ipynb`](3_layer_nn_character_pred.ipynb)
· Background and paper mapping: [top-level README](../README.md)

## Pipeline

```
names ──► (5 previous chars → next char) ──► C[X] ──► view(-1,50) ──► tanh(·W1+b1) ──► ·W2+b2 ──► softmax/CE
          182k / 22k / 22k examples         [B,5,10]   [B,50]          [B,500]           [B,27]
```

1. **Vocabulary**: `a`–`z` → 1–26, `.` → 0.
2. **Dataset** (`build_dataset`, `block_size = 5`): every character of every name, including the
   final `.`, is a target. The input is the 5 characters before it, left-padded with `.`. Names are
   split 80/10/10 into train/val/test *by name*, so no name appears in two splits.
3. **Parameters**: `C` 27×10, `W1` 50×500, `b1` 500, `W2` 500×27, `b2` 27, for 39,297 in total.
4. **Training**: 100k steps of mini-batch (1,024) momentum SGD.

   ```python
   loss = F.cross_entropy(logits, Ytr[idx]) + 0.001*(W1**2).sum() + 0.001**(W2**2).sum()
   v = beta*v + grad          # heavy-ball momentum
   p -= lr*v
   every 10k steps:  lr = max(lr/2, 2e-4);  beta = min(beta + 0.01, 0.95)
   ```

   Validation cross-entropy is logged every 1,000 steps.
5. **Evaluation**: validation cross-entropy **2.238** (perplexity 9.4 per character).
6. **Sampling**: start from `.....`, sample from the softmax, slide the window, and stop at `.`.

## Reading the loss curve

![loss curve](../assets/main_loss_curve.png)

- **Steps 0–5k: spikes.** The weights start from unscaled `randn`, so `tanh` units are saturated
  and logits are huge. The first-step loss of 68 is about 43 cross-entropy plus 25 L2 penalty. The
  early effective step `lr/(1−β)` ≈ 1.1 is large, and training bounces before it settles.
- **5k–60k: steady descent.** Validation loss keeps falling as `lr` is halved.
- **60k–100k: plateau.** Validation flattens at about 2.24, because `lr` is now tiny (≤ 0.0025).
- **The gap between train and validation (≈ 0.4)** is mostly *not* overfitting. The split isn't
  shuffled, and the validation names come from the rare tail of `names.txt`. A seeded re-run of this
  exact recipe on a shuffled split gets val **2.010** with a gap of only 0.12. Details are in the
  [top-level README](../README.md#things-worth-knowing-about-this-run).

## Notes on the code

| Line | What happens | Effect |
|---|---|---|
| `0.001**(W2**2).sum()` | `**` instead of `*` | the term is exactly 0, so W2 is unregularised |
| logged `loss` | includes the L2 term | train and val numbers aren't on the same footing early on |
| `random.seed(42)` with no shuffle | split follows file order | val/test are a harder, shifted distribution |
| `# loss on test set` | evaluates `Xval` | test split never scored |
| val eval inside the loop | no `torch.no_grad()` | builds an unused autograd graph every 1k steps (wasted memory) |
| `abs(loss - prev_loss) < 1e-8` | early-stop on a noisy mini-batch loss | never fires |
| best-weights block | commented out | the final weights, not the best-on-val ones, are used |

Suggested next steps:
- shuffle before splitting;
- fix `*` in the W2 term;
- Kaiming-style init scaling;
- evaluate under `torch.no_grad()`;
- keep the best-val checkpoint;
- report test once at the end.
