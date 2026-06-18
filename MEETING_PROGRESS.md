# Project Progress — Meeting by Meeting

**Project:** Automatic Speech Recognition for Low-Resource Telugu  
**Students:** Rahul Raj, Sai Srikar Devasani  
**Supervisor:** Dongyun Nie | **Second Reader:** Silvana MacMahon  

---

| # | Date | Type | Key Task Set | What Was Done | Proof / Evidence |
|---|---|---|---|---|---|
| M1 | 2025-09-24 | Progress | Narrow scope: identify goal, dataset, novelty, 3 research questions, 5+ related papers | Scoped project to low-resource Telugu ASR; identified MSCIL as primary dataset; defined research questions around KenLM + data scaling | `PROJECT_PRD.md` — research questions, dataset table, related papers section |
| M2 | 2025-10-15 | Progress | Identify top 3 ASR models; find multi-speaker datasets (≥20 participants, ≥1h); replicate top paper | Identified wav2vec2-XLS-R, Whisper, MMS as top-3 models; sourced Shrutilipi (AI4Bharat) and OpenSLR-66 as additional datasets | `PROJECT_PRD.md` — model table (Section 6); dataset table (Section 3) |
| M3 | 2025-11-12 | Progress | Run wav2vec2 and wav2vec2-300M starting from 5 hours to get early output | Initial pipeline prototyped and validated end-to-end; baseline training begun on MSCIL | `02_model_pipeline_prototyping.ipynb` — CTC inference working; `03_full_scale_training.ipynb` — baseline started |
| M4 | 2025-11-26 | Approval | Project approval presentation — novelty, methodology, references | **Project approved.** Feedback: clarify novelty, use peer-reviewed refs beyond ArXiv | `EXPERIMENT_REPORT.md` — novelty framed as learning curves + multi-model comparison gap (first for Telugu); `PROJECT_PRD.md` Section 2 — research gap table |
| M5 | 2026-01-29 | Progress | 10–15h of data collected and processed; address character interpretation challenges | MSCIL cleaned: 35,426 samples (~38h); baseline trained → **WER 45.27%, CER 9.73%**; identified `mask_feature_prob` misconfiguration as key issue | `01_data_exploration_and_cleaning.ipynb` — EDA output: 40h raw, 35,426 retained; `03_full_scale_training.ipynb` — baseline result; `EXPERIMENT_REPORT.md` — root-cause analysis |
| M6 | 2026-02-17 | Progress | Train/test accuracy at steps 1800/2000/2200; analyse results | EXP-1 hyperparameters applied (LR 3e-4→1e-4, mask_feat_prob 0.004→0.1); multi-dataset built (91,951 samples, ~197h); EXP-1 results: **Greedy 43.12% WER → KenLM 29.69% WER** | `04_multi_dataset_preparation.ipynb` — 91,951 samples confirmed; `05_improved_full_training.ipynb` — training log + KenLM eval output |
| M7 | 2026-03-27 | Progress | Deep analysis of results; apply to second dataset; expand dataset (no copyright issues) | MSCIL-only control run: **Greedy 46.28% → KenLM 28.44%**; learning curve notebook (07) created for 10%/25%/50%/100% fractions; Shrutilipi (CC BY 4.0) and OpenSLR-66 (CC BY-SA 4.0) added | `05b_mscil_only_training.ipynb` — MSCIL-only results; `07_wav2vec2_learning_curves.ipynb` — parameterised by DATA_FRACTION; `README.md` — dataset license section |
| M8 | 2026-04-17 | Ethics | Ethics and GDPR review | **Ethics approved** — anonymised publicly available datasets; no personal data collected | All datasets are public: MSCIL (institutional), Shrutilipi (CC BY 4.0), OpenSLR-66 (CC BY-SA 4.0), FLEURS (Google, public) |
| M9 | 2026-05-12 | Progress | Literature review submission | **Literature Review scored 8/10** — strong synthesis, clearly identified research gaps, up-to-date references | `ASR_Literaure_Review.pdf` — submitted; feedback: well-organised, strong state-of-the-art coverage |

---

## Results Summary (as of M9)

| Configuration | Dataset | Greedy WER | KenLM WER | Improvement |
|---|---|---|---|---|
| Baseline EXP-0 | MSCIL only (~35k, ~38h) | 45.27% | — | — |
| EXP-1 | MSCIL only (~35k, ~38h) | 46.28% | **28.44%** | −17.8pp (39% rel.) |
| EXP-1 | Multi-dataset (~91k, ~197h) | 43.12% | **29.69%** | −13.4pp (31% rel.) |

**Best result so far:** 29.69% WER  
**Published ceiling (reference):** vasista22 — 9.65% WER on FLEURS te_in (Whisper-large-v2, much more data)

---

## Next Steps (post M9)

| Notebook | Purpose | Status |
|---|---|---|
| `07_wav2vec2_learning_curves.ipynb` | wav2vec2 at 10%/25%/50%/100% MSCIL + FLEURS te_in eval | Ready to run |
| `06.ipynb` | Whisper-large-v3 LoRA fine-tuning | Planned |
| `08_whisper_small_learning_curves.ipynb` | Whisper-small learning curves | To build |
| `10_mms_zeroshot.ipynb` | MMS-300M zero-shot baseline | To build |
| `11_analysis_and_plots.ipynb` | Final plots and thesis figures | To build |

---

## Key Communications

### Email to Supervisor — Research Direction Proposal (~March 2026)

> Sent after M7, summarising completed experiments and proposing the shift to a learning curve + multi-model research question.

---

Hi Robin,

Hope you're well — quick progress update from our end.

**What we've done:**

- Merged three public Telugu datasets (MSCIL, Shrutilipi, OpenSLR-66) into a single corpus of 91,951 samples (~197 hours)
- Fine-tuned wav2vec2-xls-r-300m with corrected SpecAugment regularisation (the baseline had `mask_feature_prob` effectively disabled)
- Applied KenLM 5-gram shallow fusion at decode time — no retraining required

**Results:**

| Configuration | Greedy WER | KenLM WER |
|---|---|---|
| Baseline (MSCIL only) | 45.27% | — |
| EXP-1 (MSCIL only, ~38h) | 46.28% | 28.44% |
| EXP-2 (Multi-dataset, ~197h) | 43.12% | **29.69%** |

*MSCIL = Microsoft Speech Corpus Indian Languages*

**Best so far: 29.69% WER — ~34% relative improvement over baseline.**

**Direction we'd like your thoughts on:**

We feel that purely chasing WER doesn't add much to the field. We'd like to shift focus to a more meaningful research question:

> *What is the minimum viable training data size for practical Telugu ASR, and how does this trade-off differ across transformer architectures?*

Would appreciate your take before we commit to the next set of experiments.

Best regards,
Sai Srikar Devasani
