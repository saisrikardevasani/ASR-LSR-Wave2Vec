# Telugu ASR — MSc Thesis Project Reference Document
**Student:** Sai Srikar Devasani  
**Programme:** MSc Artificial Intelligence, Dublin City University (1-year)  
**Last Updated:** 2026-05-20  
**Status:** Supervisor confirmation pending for full experiment direction (Section 4 onwards)

---

## 1. What This Project Is

Fine-tuning transformer ASR models on Telugu using ~35,000 samples (MSCIL dataset, ~90h) and measuring how far a small dataset + language model post-processing can take us.

**Current confirmed work** (supervisor-approved):
- Baseline training with `facebook/wav2vec2-xls-r-300m` on MSCIL data
- EXP-1 hyperparameter fixes (mask_feature_prob, LR, warmup)
- KenLM 5-gram shallow fusion to measure how much a language model improves greedy CTC decoding
- Multi-dataset extension (MSCIL + Shrutilipi, ~91k samples) to measure data scaling

**Proposed direction** (awaiting supervisor confirmation):
- Learning curve study: 4 data fractions × multiple models on FLEURS te_in benchmark
- Multi-model comparison (wav2vec2, Whisper, MMS) on the same Telugu benchmark
- This would be the Telugu equivalent of Nahabwe et al. 2025 — a confirmed research gap

### Core Research Question (current)
> *"How much does KenLM shallow fusion improve Telugu CTC-ASR, and how does this interact with training data size?"*

### Extended Research Question (proposed — pending supervisor)
> *"What is the minimum viable training data size and model configuration for practical Telugu ASR, and how does this trade-off differ across transformer architectures?"*

---

## 2. Research Gap — Confirmed

| Paper | Models | Telugu WER | Learning Curves? |
|---|---|---|---|
| Satla & Shieh 2025 (IIETA) | wav2vec2, Whisper, HuBERT | 9.8% / 12.5% | No — 7h only, no data fraction study |
| Jain & Bhowmick 2025 (EURASIP) | wav2vec2, W2V-BERT, Whisper-small | In paper (paywalled) | No |
| Tripathi et al. 2024 (arXiv) | Whisper-medium (prompt-tuned) | 23.68% on Kathbath | No |
| Vistaar INTERSPEECH 2023 | IndicWhisper, IndicWav2Vec | In 59-benchmark table | No |
| vasista22 (HuggingFace, no paper) | Whisper-large-v2 | **9.65% on FLEURS te_in** | No |
| **Nahabwe et al. 2025** | Whisper, XLS-R, MMS, W2V-BERT | African langs (Swahili, Hausa…) | **YES — 1h to 400h** |

**The gap (no paper has done all three for Telugu):**
1. Multi-model comparison on the same Telugu benchmark in one controlled study
2. Learning curves (varying training data) for Telugu ASR
3. Both combined

This thesis fills all three gaps simultaneously.

**Upper bound reference:** vasista22's 9.65% WER on FLEURS te_in is the published ceiling — trained on much more data including FLEURS train itself. We cite this as the upper bound and show that our contribution is understanding the *efficiency curve* that leads there, not just matching the ceiling.

---

## 3. Datasets

### Primary Training Data

| Dataset | Samples | Hours | Language | Source |
|---|---|---|---|---|
| MSCIL Telugu | ~35,426 clips | ~90h | Telugu (read speech) | `./telugu_asr_clean_dataset/` |
| Shrutilipi (ai4bharat/Shrutilipi, config `te`) | ~56,000 clips | ~160h | Telugu (read speech) | HuggingFace |
| Multi-dataset combined | ~91,951 clips | ~250h | Telugu | `./telugu_multi_dataset/` |

### Evaluation Benchmark (HELD-OUT — never trained on)
- **FLEURS te_in** (`google/fleurs`, config `te_in`) — standardised, used across all published Telugu ASR papers
- Size: ~1,200 test samples
- Used for ALL model comparisons throughout this project

### Data Processing
- Cleaned: removed clips with empty transcriptions, audio length < 0.5s or > 15s, non-Telugu characters
- Resampled to 16 kHz
- Processor/tokeniser: `./telugu_wav2vec2_processor/` (67-token Telugu vocab) and `./telugu_wav2vec2_processor_v2/` (multi-dataset vocab)

---

## 4. The Experiment Matrix

Every cell evaluates on **FLEURS te_in**. Rows are models, columns are training data fractions of MSCIL (~35k clips).

| Model | 10% (~3.5k) | 25% (~8.8k) | 50% (~17.7k) | 100% MSCIL (~35k) | 100% Multi (~91k) | Runner |
|---|---|---|---|---|---|---|
| **wav2vec2-XLS-R-300M** | planned | planned | planned | ✅ done (05b) | ✅ done (05) | NB 08 auto |
| **Whisper-small** | planned | planned | planned | planned | — | NB 08 auto |
| **Whisper-medium** | — | — | planned | planned | — | NB 08 auto |
| **MMS-300M** | zero-shot | — | — | adapter FT | — | NB 08 auto |
| **vasista22/whisper-telugu-large-v2** | — | — | — | — | planned (ceiling) | NB 08 auto |
| **Whisper-large-v3 LoRA** | — | — | — | optional | — | NB 06 |

**NB 08 (`08_multi_model_learning_curve_runner.ipynb`) is the new automated runner** — randomly shuffles the full experiment matrix, skips completed runs, trains each (model × fraction) sequentially, and updates a live learning curve plot after every run.

### What "novel" means for each row
- **wav2vec2 curves**: First published learning curve for Telugu with wav2vec2
- **Whisper-small curves**: First published learning curve for Telugu with Whisper
- **Whisper-medium**: First controlled Telugu comparison vs wav2vec2 on FLEURS te_in
- **MMS zero-shot**: First published zero-shot MMS Telugu WER on FLEURS te_in
- **Whisper-large-v3 LoRA**: Demonstrates LoRA feasibility for low-resource Telugu ASR

### KenLM shallow fusion (for wav2vec2 rows)
Each wav2vec2 result is reported twice: greedy decoding WER and KenLM 5-gram WER. The delta shows language model benefit as a function of training data size — this is an additional novel finding.

---

## 5. Current Progress

### Completed ✅

| Notebook | Purpose | Key Result |
|---|---|---|
| `01_data_exploration_and_cleaning` | EDA, clean MSCIL | 35,426 clean clips, 67-token vocab |
| `02_model_pipeline_prototyping` | Sanity check pipeline | End-to-end CTC inference working |
| `03_full_scale_training` | Baseline training | **WER 45.27%, CER 9.73%** |
| `04_multi_dataset_preparation` | Merge MSCIL + Shrutilipi | 91,951 clips → `./telugu_multi_dataset/` |
| `05_improved_full_training` | EXP-1 on multi-dataset | **Greedy 43.12% → KenLM 29.69%** (13.4% absolute improvement) |

### Running ⏳

| Notebook | Purpose | Status |
|---|---|---|
| `05b_mscil_only_training` | **KenLM test on MSCIL-only data** — trains wav2vec2 on 31,883 samples, then measures greedy WER → KenLM WER improvement. One-off experiment to confirm KenLM value before deciding next direction. | Step 9,244/49,850, Epoch 9.27/50, WER 49.3% and falling, no overfitting |

### Created, awaiting supervisor confirmation 🔜

| Notebook | Purpose | Blocked on |
|---|---|---|
| `06.ipynb` | Whisper-large-v3 LoRA fine-tuning | Supervisor sign-off on direction |
| `07_wav2vec2_learning_curves.ipynb` | wav2vec2 at 10%/25%/50%/100% MSCIL + FLEURS te_in eval | Supervisor sign-off on direction |

### To Build (if supervisor approves learning curve direction) 📋

| Notebook | Purpose |
|---|---|
| `08_whisper_small_learning_curves.ipynb` | Whisper-small at 4 data fractions |
| `09_whisper_medium.ipynb` | Whisper-medium at 50% + 100% |
| `10_mms_zeroshot.ipynb` | MMS-300M zero-shot Telugu evaluation |
| `11_analysis_and_plots.ipynb` | Learning curve plots, comparison tables, thesis figures |

---

## 6. Technical Stack

### Models
| Model | Params | Architecture | Fine-tune method | Notes |
|---|---|---|---|---|
| `facebook/wav2vec2-xls-r-300m` | 300M | CNN + Transformer encoder, CTC head | Full fine-tune (CNN frozen) | Primary model — learning curves done |
| `openai/whisper-small` | 244M | Encoder-Decoder, cross-attention | Full fine-tune | Learning curve study (NB 08) |
| `openai/whisper-medium` | 307M | Encoder-Decoder, cross-attention | Full fine-tune | 50%+100% only (NB 08) |
| `facebook/mms-300m` | 300M | wav2vec2-based, 1,100-language | Zero-shot → adapter FT (`model.load_adapter("tel")`) | Zero-shot baseline + adapter run (NB 08) |
| `vasista22/whisper-telugu-large-v2` | 1.55B | Encoder-Decoder | Continued fine-tune (already Telugu-trained) | Ceiling model — 100% multi only (NB 08) |
| `openai/whisper-large-v3` | 1.54B | Encoder-Decoder | LoRA (r=32, α=64, target q_proj/v_proj) | Optional — if GPU time permits (NB 06) |

**Note on WhisperFlow:** WhisperFlow (MobiSys 2025) is a streaming inference optimizer for existing Whisper models, not a new fine-tuneable model. Not applicable to this training study.

### Infrastructure
- **GPU**: NVIDIA RTX A6000 (51 GB VRAM) on shared university server
- **Memory-adaptive batch sizing**: Auto-selects BATCH_SIZE/GRAD_ACCUM based on `torch.cuda.get_device_properties(0).total_memory`
- **Precision**: FP16 for wav2vec2, BF16 for Whisper (Ampere+ architecture)
- **Gradient checkpointing**: Enabled for all models to reduce peak VRAM

### Key Hyperparameters (EXP-1 — the fixed config across all wav2vec2 runs)
```python
LR                   = 1e-4         # was 3e-4 in baseline — key stability fix
WARMUP               = 1000         # ~1 full epoch warmup
EPOCHS               = 50           # early stopping gates actual end
MASK_TIME_PROB       = 0.1
MASK_FEAT_PROB       = 0.1          # THE key fix — was 0.004 (effectively off)
ATTENTION_DROPOUT    = 0.1
HIDDEN_DROPOUT       = 0.1
EARLY_STOP_PATIENCE  = 8
EARLY_STOP_THRESHOLD = 0.005
```

### KenLM (for wav2vec2 decoding)
- 5-gram language model built from training set transcripts using `kenlm/build/bin/lmplz`
- Decoded with `pyctcdecode.build_ctcdecoder(vocab_list, arpa_path, alpha=0.5, beta=1.0)`
- Binary confirmed present at `./kenlm/build/bin/lmplz`

---

## 7. Expected Results

### wav2vec2-XLS-R-300M (based on runs so far)
| Data | Expected Greedy WER | Expected KenLM WER |
|---|---|---|
| 10% (~3.5k clips) | 65–75% | 55–65% |
| 25% (~8.8k clips) | 55–65% | 45–55% |
| 50% (~17.7k clips) | 48–58% | 38–48% |
| 100% MSCIL (~35k) | 40–48% | 30–40% |
| 100% Multi (~91k) | **43.12% (actual)** | **29.69% (actual)** |

The learning curve shape will show **diminishing returns** — each data doubling gives smaller WER improvement. This is the primary finding for the wav2vec2 chapter.

### Whisper-small (predicted from Nahabwe et al. patterns)
- Should outperform wav2vec2 at low data fractions (encoder-decoder is more data-efficient due to language model built in)
- At 100% MSCIL: expected ~35–42% WER without KenLM (no KenLM needed — beam search handles LM internally)

### MMS-300M Zero-shot
- Expected: 60–80% WER (zero-shot, no Telugu fine-tuning)
- Value: establishes the zero-shot baseline; shows how much fine-tuning is worth

### Whisper-large-v3 LoRA
- Expected: 25–35% WER at 100% multi-dataset (best result in the study)
- This is the ceiling model — demonstrates what's achievable with the largest architecture + most data

---

## 8. What Gets Published / Submitted

### Thesis Chapters
1. **Literature Review** — Survey of Telugu ASR, low-resource ASR methods, learning curve studies (Nahabwe et al. as template)
2. **Data & Baseline** — MSCIL dataset, preprocessing, wav2vec2 baseline (45.27% → 29.69% with KenLM)
3. **Experiment Chapter 1: Learning Curves** — wav2vec2 and Whisper-small at 4 data fractions on FLEURS te_in
4. **Experiment Chapter 2: Model Comparison** — All 5 models at 100% MSCIL on FLEURS te_in
5. **Experiment Chapter 3: KenLM Analysis** — Does LM shallow fusion help equally at all data sizes?
6. **Discussion** — Efficiency curve shape, practical recommendation for new Telugu ASR practitioners, limitations

### Target Conference
**INTERSPEECH 2026** — 4-page paper on the learning curve findings (wav2vec2 + Whisper-small curves on FLEURS te_in). This is the most directly publishable result. Submission deadline typically March 2026 — feasible if experiments complete by January 2026.

---

## 9. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| GPU runs out of time / quota | Medium | High | wav2vec2 runs are the core; Whisper-medium/large are "bonus" — thesis passes without them |
| WER never beats 40% on MSCIL-only | Low | Low | The research question is about the *shape* of the curve, not hitting a specific WER. Even 50% WER at 10% data vs 43% at 100% is a valid learning curve |
| Whisper OOM on shared GPU | Low | Low | LoRA reduces Whisper-large-v3 trainable params to ~4M; optimizer states ~40 MB. Whisper-small has no OOM risk at all |
| FLEURS te_in evaluation fails | Low | Medium | FLEURS is a HuggingFace dataset — network-dependent. Cache locally: `load_dataset("google/fleurs", "te_in", split="test")` and save to disk |
| Contribution too similar to Nahabwe et al. | Low | Low | Telugu is a genetically distinct language (Dravidian vs Afroasiatic/Niger-Congo); FLEURS is the standard benchmark; the model selection is different. The parallel is a *strength* (it's a replication + extension to a new language family) |

---

## 10. Key Files Reference

```
ASR_LRL_MSCIL_FTL/
├── PROJECT_PRD.md                        ← this file
├── EXPERIMENT_REPORT.md                  ← EXP-1/2/3 hyperparameter rationale
├── vocab.json                            ← 67-token Telugu CTC vocabulary
│
├── 01_data_exploration_and_cleaning.ipynb  ✅ done
├── 02_model_pipeline_prototyping.ipynb     ✅ done
├── 03_full_scale_training.ipynb            ✅ done — baseline 45.27% WER
├── 04_multi_dataset_preparation.ipynb      ✅ done — 91,951 clips
├── 05_improved_full_training.ipynb         ✅ done — 43.12% → 29.69% KenLM
├── 05b_mscil_only_training.ipynb           ⏳ running — MSCIL-only baseline
├── 06.ipynb                               🔜 Whisper-large-v3 LoRA
├── 07_wav2vec2_learning_curves.ipynb       🔜 wav2vec2 at 4 data fractions
│   [08_whisper_small_learning_curves]      📋 to build
│   [09_whisper_medium]                     📋 to build
│   [10_mms_zeroshot]                       📋 to build
│   [11_analysis_and_plots]                 📋 to build
│
├── telugu_asr_clean_dataset/              MSCIL ~35k clips (primary)
├── telugu_multi_dataset/                  MSCIL + Shrutilipi ~91k clips
├── telugu_wav2vec2_processor/             Processor v1 (MSCIL vocab)
├── telugu_wav2vec2_processor_v2/          Processor v2 (multi-dataset vocab)
├── kenlm/build/bin/lmplz                  ✅ confirmed working
│
└── results/                               all JSON result files go here
    ├── wav2vec2_mscil_only.json           ← 05b output (pending)
    ├── wav2vec2_multi_full.json           ← 05 output (pending save)
    └── wav2vec2_lc_{source}_{frac}.json  ← 07 outputs (pending)
```

---

## 11. Immediate Next Steps (in order)

1. **Wait for `05b` to finish** — will produce MSCIL-only greedy WER + KenLM WER. Expected completion: ~12h from step 9,244. This gives the 100% MSCIL data point for the learning curve.

2. **Run `07_wav2vec2_learning_curves.ipynb`** — 4 sequential runs (10%/25%/50% MSCIL + FLEURS eval each). Each run is ~4–8h. Can be left running overnight.

3. **Build `08_whisper_small_learning_curves.ipynb`** — same structure as 07 but for Whisper-small. This is the second model in the experiment matrix and the most directly publishable comparison.

4. **Run `06.ipynb`** (Whisper-large-v3 LoRA) — requires fresh kernel (no zombie CUDA processes). Produces the ceiling WER for the model comparison table.

5. **Build `10_mms_zeroshot.ipynb`** — short notebook, just loads `facebook/mms-300m`, runs inference on FLEURS te_in, reports WER. ~1h to write, ~1h to run.

6. **Build `11_analysis_and_plots.ipynb`** — reads all `./results/*.json`, generates learning curve plots, comparison tables, and thesis-ready figures.

---

## 12. The Single Most Important Number

**29.69% WER** — achieved in notebook 05 with wav2vec2-XLS-R-300M + KenLM 5-gram on 91,951 samples (MSCIL + Shrutilipi).

This is the anchor result. Everything else in the thesis is measured relative to it: smaller data sizes show how much data matters, smaller models show how much architecture matters, and the learning curves show the efficiency trade-off between them.

The upper bound (vasista22's 9.65% on FLEURS te_in) is reachable with significantly more data and Whisper-large-v2. We are not trying to match it — we are mapping the path towards it.

---

*This document should be included in the conversation context for any future Claude session working on this project.*
