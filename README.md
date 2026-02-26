# Low-Resource ASR for Indian Languages using Self-Supervised Learning 🎙️

**[Work in Progress]** Fine-tuning self-supervised speech models (Wav2Vec2, WavLM) for Automatic Speech Recognition on low-resource Indian languages using Microsoft's Indian Language Speech Corpus. Focus on Telugu with experiments on data subsets.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗_Transformers-4.35.0-yellow.svg)](https://huggingface.co/transformers/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-In_Progress-yellow.svg)]()

## 🎯 Project Overview

This research project explores effective ASR techniques for low-resource Indian languages through:
- **Self-supervised pre-trained models**: Wav2Vec2, WavLM, and variants
- **Transfer learning**: Leveraging multilingual pre-training for limited data scenarios
- **SpecAugment**: Forcing contextual understanding over pattern memorization
- **Dataset scaling experiments**: Testing performance across different data sizes (1h, 5h, 10h+)

**Current Focus**: Telugu language subset from Microsoft Speech Corpus

## 🔬 Methodology

### 1. Self-Supervised Learning Approach

The project leverages pre-trained transformer-based speech models:

#### Model Candidates
- **Wav2Vec2-XLSR-53** (facebook/wav2vec2-xls-r-300m)
  - 300M parameters, pre-trained on 50+ languages
  - Strong multilingual acoustic representations
  
- **Wav2Vec2-XLS-R-1B** (facebook/wav2vec2-xls-r-1b)
  - 1B parameters, better for complex languages
  
- **WavLM** (microsoft/wavlm-large)
  - Enhanced pre-training with denoising
  - Better for noisy environments
  
- **IndicWav2Vec** (ai4bharat/indicwav2vec_v1_telugu)
  - Specifically pre-trained on Indian languages

### 2. Fine-Tuning Strategy

```
┌─────────────────────────────────────┐
│   Pre-trained Model                 │
│   (50+ languages, 50k+ hours)       │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│   Freeze Feature Encoder            │
│   (Keep acoustic knowledge)         │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│   Fine-tune Transformer + CTC Head  │
│   (Learn Telugu-specific patterns)  │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│          Telugu Corpus              │
│   + SpecAugment (masking)           │
└─────────────────────────────────────┘
```

**Key Technique**: Frozen feature encoder preserves pre-trained acoustic representations while fine-tuning only language-specific layers.

### 3. Training Pipeline

#### Data Preprocessing
```python
1. Load audio files (16kHz sampling rate)
2. Text normalization:
   - Keep Telugu Unicode characters (U+0C00-U+0C7F)
   - Remove punctuation and special characters
   - Replace spaces with word delimiter '|'
3. Create vocabulary from training data
4. Tokenize transcriptions
```

#### Audio Processing
```python
1. Load with torchaudio (efficient C++ backend)
2. Resample to 16kHz if needed
3. Apply feature extraction (80-dim log-mel filterbanks)
4. Dynamic padding to batch
```

#### Training Configuration
```python
# Core settings
batch_size = 4
gradient_accumulation = 2  # Effective batch: 8
learning_rate = 3e-4
num_epochs = 10
fp16 = True  # Mixed precision

# SpecAugment (key for understanding)
mask_time_prob = 0.05      # Mask 5% of time steps
mask_feature_prob = 0.05   # Mask 5% of features

# CTC Loss
ctc_loss_reduction = "mean"
ctc_zero_infinity = True   # Handle infinite loss
```

### 4. Understanding vs Memorization

**Challenge**: With limited data, models can memorize patterns instead of learning

**Solution - SpecAugment**:
- Randomly masks portions of audio during training
- Forces model to infer from incomplete information
- Learns contextual relationships, not exact patterns
- Result: Better generalization to unseen data

```
Original Audio:     [=============================]
After SpecAugment:  [====  ===  =======    ======]
                          ↓
                   Model must predict
                   missing segments
                   from context
```

### 5. CTC Loss for Sequence-to-Sequence Mapping

**Why CTC?**
- No need for frame-level alignment (expensive for low-resource)
- Handles variable-length sequences naturally
- Standard for speech recognition with limited data

**How it works**:
```
Audio frames:  [a] [a] [a] [b] [-] [c] [c] [-] [d]
                ↓   CTC collapse
Output text:   "abcd"
```

## 📊 Experimental Design

### Dataset Scaling Study

| Subset | Size | Expected Use Case |
|--------|------|-------------------|
| Small | ~1 hour (800 samples) | Quick experimentation |
| Medium | **~5 hours (4000 samples)** | **Current focus** |
| Large | ~10 hours (8000 samples) | Better performance |
| Full | 44,882 samples | Production model |

### Metrics

**Primary Metrics**:
- **Character Error Rate (CER)**: More meaningful for syllabic scripts like Telugu
- **Word Error Rate (WER)**: Word-level accuracy

**Benchmark Expectations** (based on literature):
| Data | Expected CER | Expected WER |
|------|-------------|--------------|
| 1h | 40-50% | 50-60% |
| 5h | 25-35% | 35-45% |
| 10h | 20-30% | 30-40% |

### Evaluation Strategy
1. **Validation set**: For hyperparameter tuning during training
2. **Test set**: For final evaluation (completely unseen)
3. **Greedy decoding**: Baseline (argmax of logits)
4. **Beam search + LM**: Advanced (with language model - future work)

## 🛠️ Technical Stack

```python
# Core Framework
PyTorch 2.0+
HuggingFace Transformers 4.35.0
HuggingFace Datasets 2.14.0

# Audio Processing
torchaudio (C++ backend)
librosa (backup)

# Metrics
evaluate (HuggingFace)
jiwer (WER/CER calculation)

# Environment
Kaggle
Python 3.8+
```

## 📁 Project Structure

```
├── notebooks/
│   ├── kaggle_training.ipynb          # Main training notebook
│   └── data_exploration.ipynb         # Dataset analysis
│
├── src/
│   ├── data_processing.py             # Text cleaning, vocab creation
│   ├── model.py                       # Model configuration
│   ├── training.py                    # Training loop
│   └── evaluation.py                  # Metrics and evaluation
│
├── configs/
│   ├── wav2vec2_config.yaml           # Wav2Vec2 settings
│   ├── wavlm_config.yaml              # WavLM settings
│   └── training_config.yaml           # Training hyperparameters
│
├── scripts/
│   ├── fix_dependencies.sh            # Environment setup
│   ├── prepare_data.py                # Dataset preparation
│   └── run_training.sh                # Training launcher
│
├── docs/
│   ├── KAGGLE_SETUP.md                # Cloud training guide
│   ├── TROUBLESHOOTING.md             # Common issues
│   └── METHODOLOGY.md                 # Detailed methodology
│
└── results/
    └── experiments/                    # Training logs and metrics
```

## 🔍 Current Experiments

### Phase 1: Baseline (In Progress)
- [x] Environment setup (dependencies resolved)
- [x] Data preprocessing pipeline
- [x] Vocabulary creation
- [x] Model initialization
- [x] Training on 5-hour subset
- [ ] Baseline evaluation
- [ ] Error analysis

### Phase 2: Model Comparison (Planned)
- [ ] Wav2Vec2-XLSR-53 vs WavLM
- [ ] Different masking rates (SpecAugment tuning)
- [ ] Learning rate experiments
- [ ] Frozen vs unfrozen feature encoder

### Phase 3: Advanced Techniques (Future)
- [ ] Language model integration (KenLM)
- [ ] Beam search decoding
- [ ] Data augmentation (speed, noise)
- [ ] Multi-task learning

### Phase 4: Scaling (Future)
- [ ] Full dataset training
- [ ] Other Indian languages (Hindi, Tamil)
- [ ] Cross-lingual transfer learning

## 🎯 Research Questions

1. **How much data is needed?**
   - Can 5 hours achieve usable ASR performance?
   - Diminishing returns vs data size?

2. **Which model is best?**
   - Wav2Vec2 vs WavLM vs IndicWav2Vec?
   - Parameter efficiency vs accuracy?

3. **Does SpecAugment help?**
   - Quantify understanding vs memorization
   - Optimal masking rates?

4. **Training stability**
   - Do we observe mid-training instability (steps 2000-2400)?
   - How to mitigate?

## 📈 Preliminary Observations

### Data Quality
- Dataset: 44,882 train samples, 2,640 test samples
- Audio: Clean studio recordings, 16kHz
- Transcriptions: Telugu script, variable length (1-260 chars)
- Vocabulary: 67 unique characters (including special tokens)

### Training Behavior
- **Dependency issues**: Resolved pyarrow==22.0.0 incompatibility
- **Memory requirements**: ~10GB GPU for batch_size=4
- **Processing time**: ~11 minutes for 44k samples (audio loading)
- **Training time**: ~3 hours for 10 epochs on 5h subset (P100 GPU)

### Initial Results (5-hour subset, 10 epochs)
```
Epoch | Loss  | CER    | Notes
------|-------|--------|------------------
1-2   | High  | ~100%  | Random predictions
3-5   | ↓     | ~50%   | Learning patterns
6-8   | ↓     | ~30%   | Convergence
9-10  | Stable| ~29%   | Plateau
```

**Key Observation**: Model shows clear learning progression without catastrophic forgetting or mid-training instability.

## 🔧 Technical Challenges

### Solved
- ✅ **PyArrow compatibility**: datasets==2.14.0 incompatible with pyarrow==22.0.0
  - Solution: Downgrade to pyarrow==12.0.0
- ✅ **Memory optimization**: Sequential processing instead of multiprocessing
- ✅ **Mixed precision**: FP16 for faster training

### In Progress
- ⚙️ **WER metric**: Word-level tokenization issues with Telugu script
- ⚙️ **Optimal hyperparameters**: Learning rate, batch size tuning
- ⚙️ **Checkpoint management**: Best model selection strategy

### Future
- 🔲 **Language model integration**: KenLM decoder for better WER
- 🔲 **Real-world audio**: Domain adaptation to noisy environments
- 🔲 **Deployment**: Model compression and optimization

## 📚 Background & Related Work

### Why Self-Supervised Learning?
Traditional ASR requires:
- Large labeled datasets (1000+ hours)
- Forced alignment (expensive annotation)
- Language-specific acoustic models

Self-supervised models solve this:
- Pre-trained on unlabeled audio (cheap)
- Learn universal acoustic representations
- Transfer to any language with small labeled data

### Key Papers
- Wav2Vec2: [Baevski et al., 2020](https://arxiv.org/abs/2006.11477)
- XLSR-53: [Conneau et al., 2020](https://arxiv.org/abs/2006.13979)
- WavLM: [Chen et al., 2021](https://arxiv.org/abs/2110.13900)
- IndicWav2Vec: [Javed et al., 2021](https://arxiv.org/abs/2111.03945)

### Telugu ASR Challenges
- **Low-resource**: Limited publicly available data
- **Syllabic script**: Character-level vs word-level metrics
- **Dialectal variation**: Multiple regions, accents
- **Code-mixing**: Telugu-English mixing in conversational speech

## 🚀 Getting Started

### Prerequisites
```bash
pip install torch torchaudio
pip install transformers==4.35.0
pip install datasets==2.14.0
pip install evaluate jiwer
pip install pyarrow==12.0.0  # Important: specific version
```

### Quick Start
```bash
# Clone repository
git clone https://github.com/[your-username]/telugu-asr.git
cd telugu-asr

# Install dependencies
pip install -r requirements.txt

# Run training (Kaggle environment)
# Upload notebook to Kaggle and run
# OR
python src/training.py --config configs/wav2vec2_config.yaml
```

### For Kaggle
1. Upload dataset to Kaggle
2. Create new notebook
3. Add dataset to notebook
4. Run cells from `kaggle_training.ipynb`
5. Training takes ~28 hours on T4x2 GPU

## 📊 Expected Deliverables

### Code
- [x] Complete training pipeline
- [x] Data preprocessing utilities
- [x] Model configuration scripts
- [ ] Evaluation and inference code
- [ ] Deployment-ready model

### Documentation
- [x] Setup instructions
- [x] Training guide
- [x] Troubleshooting guide
- [ ] API documentation
- [ ] Model card

### Research
- [ ] Comparative study (multiple models)
- [ ] Error analysis
- [ ] Ablation studies (SpecAugment, frozen encoder)
- [ ] Scaling experiments (1h, 5h, 10h, full)
- [ ] Technical report/paper

## 🤝 Contributing

This is an academic research project. Contributions welcome:
- Model experiments (different architectures)
- Hyperparameter tuning
- Additional languages
- Documentation improvements
- Bug fixes

## 📝 Citation

Work in progress. Citation will be added upon completion.

## 📄 License

MIT License

## 🙏 Acknowledgments

- **Dataset**: Microsoft Speech Corpus - Indian Languages
- **Models**: Facebook AI (Wav2Vec2), Microsoft (WavLM), AI4Bharat (IndicWav2Vec)
- **Infrastructure**: Kaggle (free GPU access) & Renku-DCU in house GPU
- **Framework**: HuggingFace Transformers team

## 📧 Contact

For questions or collaboration:
- Open an issue on GitHub
- Email: saisrikardevasani@gmail.copm

---

**⚠️ Note**: This is an active research project. Results and documentation are continuously being updated as experiments progress.

**Last Updated**: February 2026
