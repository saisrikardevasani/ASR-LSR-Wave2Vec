# Telugu ASR Hyperparameter Tuning Report
**Model:** `facebook/wav2vec2-xls-r-300m` → CTC fine-tune
**Dataset:** MSCIL Telugu — 31,883 train / 3,543 eval
**Target:** WER < 40%

---

## Root-Cause Analysis of Baseline (WER 45.27%)

| Observation | Evidence | Implication |
|---|---|---|
| WER 45.27% vs CER 9.73% (5× gap) | training.log final eval | Model knows characters well; word-boundary errors dominate WER |
| Recurring anusvara `ం` artifact at word end | ~100 REF/PRED log pairs in training.log | CTC is over-predicting the most frequent Telugu character, inflating WER |
| `mask_feature_prob = 0.004` ≈ off | Cell 4 baseline config | No frequency-bin regularisation → model over-relies on specific feature dimensions |
| Training plateau at epoch 26–27 of 30 | WER stuck at 0.4527–0.4530 for last 3 evals | LR cosine decay exhausted headroom; early-stopping patience=5 triggered |
| LR = 3e-4 with only 500 warmup steps | Cell 4 baseline config | Aggressive LR with short warmup can cause noisy early updates and premature convergence |
| CNN encoder frozen throughout | `model.freeze_feature_encoder()` never lifted | Features pre-trained on 436K multilingual hours never adapted to Telugu acoustics |

---

## Experiment Table

| # | Name | LR | Warmup | mask\_time\_prob | mask\_feat\_prob | mask\_time\_len | attn\_dropout | Epochs | CNN | Patience | **WER** | **CER** | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **0** | **Baseline** | 3e-4 | 500 | 0.075 | 0.004 | 10 | 0.10 | 30 | Frozen | 5 | **0.4527** | **0.0973** | Original run; plateau at ep 26 |
| **1** | **EXP-1** ✅ | **1e-4** | **1000** | **0.1** | **0.1** | 10 | 0.10 | **50** | Frozen | **8** | *pending* | *pending* | Primary tuning run — notebook updated |
| **2** | **EXP-2** | 1e-5 | 0 | 0.1 | 0.1 | 10 | 0.10 | 20 | **Unfrozen** | 6 | *pending* | *pending* | Resume EXP-1 best ckpt; constant LR |
| **3** | **EXP-3** | 1e-4 | 1000 | **0.15** | 0.1 | **15** | **0.15** | 50 | Frozen | 8 | *pending* | *pending* | Fresh run if EXP-1/2 WER ≥ 40% |

✅ = already applied to `03_full_scale_training.ipynb`

---

## Change-by-Change Rationale

### EXP-1: Lower LR + Fix SpecAugment + Longer Warmup

| Parameter | Baseline | EXP-1 | Why |
|---|---|---|---|
| `LR` | 3e-4 | **1e-4** | Lower LR → finer weight updates in late training; prevents overshooting in the shallow loss landscape after epoch 10 |
| `WARMUP` | 500 | **1000** | With ~1,000 steps/epoch (31k samples ÷ batch 32), 500 warmup is < half an epoch. 1000 ≈ 1 full epoch warmup stabilises gradient norms before cosine decay begins |
| `MASK_FEAT_PROB` | 0.004 | **0.1** | **Biggest single fix.** Standard wav2vec2 fine-tuning uses 0.1. 0.004 is effectively off — the model over-fits specific frequency bins. 0.1 forces acoustic-feature generalisation |
| `MASK_TIME_PROB` | 0.075 | **0.1** | Slightly stronger time masking. Forces the model to predict from partial context rather than leaning on adjacent frames. Especially important for Telugu's long consonant clusters |
| `EPOCHS` | 30 | **50** | More budget. Early stopping (patience=8) ensures we stop when improvement genuinely stalls, not prematurely |
| `early_stopping_patience` | 5 | **8** | Cosine decay's long tail can still yield improvements over 6–7 consecutive eval windows. Patience=5 was too aggressive |
| `early_stopping_threshold` | 0.001 | **0.005** | 0.001 WER is noise-level at this scale (3,543 eval samples). 0.005 = ~18 word difference; meaningful |
| `OUTPUT_DIR` | `…-telugu` | **`…-telugu-exp1`** | Preserve baseline checkpoint for comparison |

**Expected WER:** ~38–42%
**Key mechanism:** The `mask_feature_prob` fix alone removes a major regularisation hole. The lower LR allows the model to settle into sharper minima without bouncing.

---

### EXP-2: Resume + Unfreeze CNN (constant 1e-5 LR)

Run only after EXP-1 converges.

| Change | Value | Why |
|---|---|---|
| Resume from EXP-1 best checkpoint | ✓ | Avoids re-learning what EXP-1 already found; adds only CNN-adaptation pass |
| Unfreeze CNN feature encoder | ✓ | XLS-R CNN was trained on 128 languages but never specifically on Dravidian vowel harmony or Telugu retroflex stops. Allowing it to adapt can improve feature quality for this phonological space |
| LR | 1e-5 | 10× lower than EXP-1 to avoid catastrophic forgetting of the Transformer layers while the CNN moves |
| Scheduler | constant | No decay — keeps a steady, predictable update size across the entire pass |

**Expected WER:** ~35–39%
**Risk:** If LR is still too high for the CNN, it can collapse representations. Mitigated by the very low 1e-5 setting.

---

### EXP-3: Stronger SpecAugment (fallback if EXP-1/2 > 40% WER)

| Change | Value | Why |
|---|---|---|
| `mask_time_prob` | 0.15 | Stronger time masking forces more context-awareness |
| `mask_time_length` | 15 | Longer spans (vs 10) create harder prediction problems, better generalisation |
| `attention_dropout` | 0.15 | Additional Transformer-level regularisation to fight word-level over-fitting |
| Fresh training | from scratch | Stronger augmentation must be present from step 0; resuming would only apply it to the last N steps |

**Expected WER:** ~37–41%
**Note:** Run as fresh training; don't resume from EXP-1.

---

## Optimal Configuration (recommended as production baseline)

Run EXP-1, then EXP-2. If combined WER < 40%, EXP-2 is the final model. Otherwise run EXP-3.

```python
# Final recommended config (Cell 4)
EPOCHS            = 50
LR                = 1e-4
WARMUP            = 1000
ATTENTION_DROPOUT = 0.1
HIDDEN_DROPOUT    = 0.1
FEAT_PROJ_DROPOUT = 0.0
MASK_TIME_PROB    = 0.1
MASK_TIME_LENGTH  = 10
MASK_FEAT_PROB    = 0.1        # THE key fix vs baseline
```

Followed by EXP-2 (resume, unfreeze CNN, LR=1e-5 constant, 20 epochs).

---

## Additional Recommendations (beyond hyperparameters)

1. **Language model shallow fusion** — A 3-gram/4-gram KenLM trained on Telugu text (Wikipedia + Common Crawl Telugu) and applied via `pyctcdecode` at decode time typically yields 3–8% absolute WER improvement on CTC models without any re-training.

2. **Anusvara over-prediction fix** — The model appends an extra `ం` at many word endings. This is a vocabulary-frequency artifact. Post-processing rule: strip trailing `ం` if it does not appear in the reference vocabulary for that word. Can be applied at inference time and would improve WER by an estimated 1–2%.

3. **Longer audio clips** — The current filter keeps clips ≤ 15 s. Clips between 10–15 s are harder for the model. Consider a secondary filter at 12 s with separate fine-tuning data splits to evaluate length sensitivity.
