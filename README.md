# Telugu ASR — Low-Resource Speech Recognition with Wav2Vec2

**MSc Artificial Intelligence Thesis Project**  
**Students:** Sai Srikar Devasani, Rahul Raj | Dublin City University  
**Research Focus:** Low-resource Telugu ASR with KenLM shallow fusion and data scaling analysis

---

## Overview

This project fine-tunes `facebook/wav2vec2-xls-r-300m` on Telugu speech data and evaluates the combined effect of:

1. **Corrected SpecAugment regularisation** — fixing a critical `mask_feature_prob` misconfiguration in the baseline
2. **Multi-dataset scaling** — merging MSCIL, Shrutilipi (AI4Bharat), and OpenSLR-66 into a 91,951-sample corpus (~197 hours)
3. **KenLM 5-gram shallow fusion** — applying a language model at decode time without any re-training

**Best result:** 29.69% WER (multi-dataset + KenLM), down from a 45.27% baseline — a 34% relative improvement.

---

## Research Questions

**Confirmed:**
> *How much does KenLM shallow fusion improve Telugu CTC-ASR, and how does this interact with training data size?*

**Proposed (pending supervisor approval):**
> *What is the minimum viable training data size for practical Telugu ASR, and how does this trade-off differ across transformer architectures?*

---

## Results

| Configuration | Dataset | Greedy WER | KenLM WER | KenLM Improvement |
|---|---|---|---|---|
| Baseline (EXP-0) | MSCIL only (~35k) | 45.27% | — | — |
| EXP-1 | MSCIL only (~35k) | 46.28% | 28.44% | −17.8pp (39% rel.) |
| EXP-1 | Multi-dataset (~91k) | 43.12% | **29.69%** | −13.4pp (31% rel.) |

**Published upper bound (reference):** vasista22, Whisper-large-v2 on FLEURS te_in — 9.65% WER (trained on significantly more data including FLEURS train itself).

---

## Datasets

| Dataset | Samples | Hours | Source |
|---|---|---|---|
| MSCIL Telugu | 35,426 | ~38h | Local — `./telugu_asr_clean_dataset/` |
| Shrutilipi (AI4Bharat `te`) | 52,077 | ~149h | HuggingFace `ai4bharat/Shrutilipi` |
| OpenSLR-66 | 4,448 | ~10h | openslr.org/66 |
| **Multi-dataset combined** | **91,951** | **~197h** | Local — `./telugu_multi_dataset/` |

**Evaluation benchmark (held-out, never trained on):**  
FLEURS te_in (`google/fleurs`, config `te_in`) — the standard Telugu ASR benchmark used across published papers.

---

## Key Improvement: EXP-1 Hyperparameter Fixes

The baseline (EXP-0) had one critical misconfiguration: `mask_feature_prob = 0.004`, which is effectively disabled. Standard wav2vec2 fine-tuning uses `0.1`. This single fix is responsible for most of the regularisation improvement.

| Parameter | Baseline | EXP-1 | Rationale |
|---|---|---|---|
| `mask_feature_prob` | 0.004 | **0.1** | **Primary fix.** Forces acoustic-feature generalisation across frequency bins |
| `learning_rate` | 3e-4 | **1e-4** | Finer updates in the shallow loss landscape after epoch 10 |
| `warmup_steps` | 500 | **1000** | ~1 full epoch warmup stabilises gradient norms before cosine decay |
| `mask_time_prob` | 0.075 | **0.1** | Stronger time masking for Telugu's long consonant clusters |
| `max_epochs` | 30 | **50** | More budget; early stopping prevents over-running |
| `early_stopping_patience` | 5 | **8** | Allows cosine decay tail to yield improvements |
| `early_stopping_threshold` | 0.001 | **0.005** | 0.001 WER is noise-level; 0.005 ≈ 18 words on 3.5k eval samples |

```python
# Final EXP-1 configuration
LR                   = 1e-4
WARMUP               = 1000
EPOCHS               = 50
MASK_TIME_PROB       = 0.1
MASK_FEAT_PROB       = 0.1    # THE fix — was 0.004 (effectively off)
ATTENTION_DROPOUT    = 0.1
HIDDEN_DROPOUT       = 0.1
EARLY_STOP_PATIENCE  = 8
EARLY_STOP_THRESHOLD = 0.005
```

---

## Project Structure

```
ASR_LRL_MSCIL_FTL/
│
├── 01_data_exploration_and_cleaning.ipynb  # EDA, MSCIL cleaning → 35,426 clips, 67-token vocab
├── 02_model_pipeline_prototyping.ipynb     # End-to-end CTC pipeline sanity check
├── 03_full_scale_training.ipynb            # Baseline training → 45.27% WER, 9.73% CER
├── 04_multi_dataset_preparation.ipynb      # Merge datasets → 91,951 clips, 69-token vocab v2
├── 05_improved_full_training.ipynb         # EXP-1 on multi-dataset → 43.12% / 29.69% KenLM ✅
├── 05b_mscil_only_training.ipynb           # EXP-1 on MSCIL-only → 46.28% / 28.44% KenLM ✅
├── 06.ipynb                               # Whisper-large-v3 LoRA fine-tuning (planned)
├── 07_wav2vec2_learning_curves.ipynb       # wav2vec2 at 10%/25%/50%/100% MSCIL + FLEURS eval
│
├── vocab.json                             # 67-token Telugu CTC vocabulary (v1, MSCIL)
├── vocab_v2.json                          # 69-token Telugu CTC vocabulary (v2, multi-dataset)
├── EXPERIMENT_REPORT.md                   # Hyperparameter tuning rationale and experiment table
├── PROJECT_PRD.md                         # Full project reference document
│
├── telugu_asr_clean_dataset/              # MSCIL ~35k clips
├── telugu_multi_dataset/                  # MSCIL + Shrutilipi + OpenSLR-66 ~91k clips
├── telugu_wav2vec2_processor/             # Processor v1 (MSCIL vocabulary)
├── telugu_wav2vec2_processor_v2/          # Processor v2 (multi-dataset vocabulary)
├── kenlm/build/bin/lmplz                  # KenLM binary for 5-gram ARPA construction
│
└── results/                              # JSON result files (created on first experiment run)
    ├── wav2vec2_mscil_only.json          # 05b output
    ├── wav2vec2_multi_full.json          # 05 output
    └── wav2vec2_lc_{source}_{fraction}.json  # 07 output per run
```

---

## Notebook Walkthrough

### `01_data_exploration_and_cleaning.ipynb`
- Loads raw MSCIL Telugu audio clips
- Filters: empty transcriptions, audio length < 1.0s or > 15s, non-Telugu characters
- Resamples all audio to 16 kHz
- Builds a 67-token Telugu CTC vocabulary (`vocab.json`)
- Output: `./telugu_asr_clean_dataset/` with 35,426 clean samples

### `02_model_pipeline_prototyping.ipynb`
- Validates the full inference pipeline: audio → feature extraction → CTC decoding → text
- Tests tokeniser round-trip encoding/decoding
- Runs a small training sanity check before committing to full-scale training

### `03_full_scale_training.ipynb`
- Trains `wav2vec2-xls-r-300m` on MSCIL with the original (baseline) hyperparameters
- CNN feature encoder frozen; all transformer layers trained
- **Result:** WER 45.27%, CER 9.73%
- Identified issues: `mask_feature_prob ≈ 0`, premature early stopping, high LR

### `04_multi_dataset_preparation.ipynb`
- Downloads Shrutilipi (AI4Bharat, HuggingFace) and OpenSLR-66
- Normalises column names, applies Telugu Unicode NFC normalisation
- Filters clips to 1.0–15.0 second duration window
- Merges into a unified 91,951-sample corpus saved as `./telugu_multi_dataset/`
- Builds a 69-token vocabulary (`vocab_v2.json`) and new processor (`telugu_wav2vec2_processor_v2/`)

### `05_improved_full_training.ipynb` — **Primary Experiment**
- Applies all EXP-1 hyperparameter fixes to the multi-dataset corpus
- Memory-adaptive batch sizing based on detected GPU VRAM
- Trains for up to 50 epochs with early stopping (patience=8)
- **Greedy decoding:** WER 43.12%, CER 8.16%
- Builds a KenLM 5-gram language model from training transcripts (`lmplz`)
- **KenLM decoding (alpha=0.5, beta=1.0):** WER **29.69%** — 13.4pp absolute improvement

### `05b_mscil_only_training.ipynb`
- Same EXP-1 configuration but trained only on MSCIL (31,883 samples)
- Isolates the effect of KenLM from multi-dataset scaling
- **Greedy:** WER 46.28% | **KenLM:** WER 28.44% (17.8pp improvement)
- Comparison shows multi-dataset helps the acoustic model more; KenLM provides larger relative gain on smaller datasets

### `07_wav2vec2_learning_curves.ipynb`
- Parameterised by `DATA_FRACTION` ∈ {0.10, 0.25, 0.50, 1.00} and `DATASET_SOURCE` ∈ {mscil, multi}
- Trains a separate model for each (fraction, source) combination using the EXP-1 config
- Evaluates on both internal validation set and held-out FLEURS te_in benchmark
- Saves structured JSON results to `./results/wav2vec2_lc_{source}_{fraction}.json`
- Generates the learning curve showing diminishing returns as data scales

---

## Technical Stack

| Component | Detail |
|---|---|
| **Base model** | `facebook/wav2vec2-xls-r-300m` (300M params, pre-trained on 436k hours, 128 languages) |
| **Fine-tuning** | CTC head, CNN encoder frozen, all transformer layers trained |
| **Decoding** | Greedy CTC + KenLM 5-gram shallow fusion via `pyctcdecode` |
| **Training framework** | HuggingFace `transformers` Trainer with `EarlyStoppingCallback` |
| **Dataset library** | HuggingFace `datasets` |
| **Language model** | KenLM `lmplz` 5-gram ARPA, built from training transcripts only |
| **Precision** | FP16 (wav2vec2), BF16 (Whisper, when applicable) |
| **GPU** | NVIDIA RTX A6000 (51 GB VRAM) |
| **Python** | 3.10+ |

### Dependencies

```bash
pip install transformers datasets evaluate jiwer torch torchaudio
pip install pyctcdecode kenlm
# KenLM binary: build from source or install kenlm Python package
```

---

## Running the Experiments

### Step 1 — Data Preparation
Run notebooks in order:
```
01_data_exploration_and_cleaning.ipynb   # produces ./telugu_asr_clean_dataset/
04_multi_dataset_preparation.ipynb       # produces ./telugu_multi_dataset/
```

### Step 2 — Training

**Multi-dataset (primary experiment):**
```
05_improved_full_training.ipynb
```

**MSCIL-only (control/comparison):**
```
05b_mscil_only_training.ipynb
```

### Step 3 — Learning Curves
Set `DATA_FRACTION` and `DATASET_SOURCE` at the top of the notebook and run:
```
07_wav2vec2_learning_curves.ipynb
```
Repeat for each fraction: 0.10, 0.25, 0.50, 1.00.

### Step 4 — KenLM Decoding
KenLM decoding is embedded in notebooks 05, 05b, and 07. The ARPA is built automatically from training transcripts using the bundled `kenlm/build/bin/lmplz` binary.

---

## Vocabulary

The Telugu CTC vocabulary covers Unicode block U+0C00–U+0C7F (Telugu script):

| Version | File | Tokens | Used in |
|---|---|---|---|
| v1 | `vocab.json` | 67 (64 Telugu chars + `\|` + `[UNK]` + `[PAD]`) | Notebooks 03, 05b |
| v2 | `vocab_v2.json` | 69 (66 Telugu chars + `\|` + `[UNK]` + `[PAD]`) | Notebooks 04, 05, 07 |

`[PAD]` is always the last token (highest ID) to serve as the CTC blank token.

---

## Planned Experiments

| Notebook | Status | Purpose |
|---|---|---|
| `06.ipynb` | Planned | Whisper-large-v3 LoRA fine-tuning — ceiling model |
| `07_wav2vec2_learning_curves.ipynb` | Created | wav2vec2 at 4 data fractions + FLEURS te_in |
| `08_whisper_small_learning_curves.ipynb` | To build | Whisper-small at 4 data fractions |
| `09_whisper_medium.ipynb` | To build | Whisper-medium at 50% + 100% MSCIL |
| `10_mms_zeroshot.ipynb` | To build | MMS-300M zero-shot Telugu WER on FLEURS te_in |
| `11_analysis_and_plots.ipynb` | To build | Learning curve plots and thesis figures from `./results/*.json` |

---

## Research Context

This work addresses a gap in the Telugu ASR literature: no published study has combined (1) multi-model comparison, (2) learning curves over varying data sizes, and (3) both together on a standard benchmark (FLEURS te_in). The closest analogue is Nahabwe et al. 2025, which performed this analysis for African languages — this thesis applies the same methodology to Telugu (Dravidian language family).

| Paper | WER on FLEURS te_in | Learning Curves? |
|---|---|---|
| Satla & Shieh 2025 | 9.8% (wav2vec2) | No — 7h only |
| vasista22 (HuggingFace) | **9.65%** (Whisper-large-v2) | No |
| Nahabwe et al. 2025 | African langs only | **Yes** — 1h to 400h |
| **This project** | 29.69% (wav2vec2 + KenLM, ~197h) | **In progress** |

---

## Citation

If you use this codebase or results, please cite:

```
Sai Srikar Devasani. "Low-Resource Telugu ASR: KenLM Shallow Fusion and Data Scaling 
with Wav2Vec2-XLS-R." MSc Thesis, Dublin City University, 2026.
```

---

## License

Academic research use. Dataset licenses apply independently:
- MSCIL: research use (institutional dataset)
- Shrutilipi: CC BY 4.0 (AI4Bharat)
- OpenSLR-66: CC BY-SA 4.0
