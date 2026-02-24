# 🎙️ Low-Resource Automatic Speech Recognition using Self-Supervised Learning

## 📌 Overview

This project focuses on building an **Automatic Speech Recognition (ASR)** system for low-resource Indian languages using state-of-the-art self-supervised speech models.

Low-resource languages often lack sufficient annotated speech data, making traditional ASR approaches ineffective. This project leverages pretrained transformer-based speech models and fine-tunes them on limited labeled datasets to improve transcription performance.

---

## 🚀 Key Features

- Fine-tuned Wav2Vec 2.0
- Experimented with WavLM
- Trained on Microsoft Indian Language Speech Corpus
- Focused on Telugu and other low-resource Indian languages
- Conducted dataset scaling experiments
- Analyzed training instability during mid-training steps
- Evaluated performance using Word Error Rate (WER)

---

## 🧠 Problem Statement

Building robust ASR systems for low-resource languages is challenging due to:

- Limited annotated speech data
- Pronunciation and dialect variability
- Data imbalance
- Overfitting on small datasets

This project investigates how self-supervised pretrained models can improve ASR performance under such constraints.

---

## 🏗️ Architecture

### 1️⃣ Pretrained Backbone
- Wav2Vec 2.0 / WavLM (Transformer-based encoders)

### 2️⃣ Fine-Tuning Strategy
- CTC (Connectionist Temporal Classification) Loss
- HuggingFace Transformers framework
- Gradient accumulation for memory efficiency

### 3️⃣ Training Configuration
- Variable dataset sizes
- Learning rate scheduling
- Mixed precision training
- Early stopping experiments

---

## 📊 Experiments

### Dataset Scaling Study

| Dataset Size | Observation |
|--------------|------------|
| Small subset | Faster convergence but high overfitting |
| Medium subset | Improved generalization |
| Larger subset | Lower WER but increased training time |

### Training Instability Analysis

Observed behavior around 2000–2400 steps:
- Temporary increase in validation loss
- Possible causes:
  - Learning rate adjustments
  - Overfitting
  - Gradient instability
  - Data distribution shifts

This analysis helped refine training strategy and hyperparameter tuning.

---

## 📈 Evaluation Metrics

- Word Error Rate (WER)
- Validation Loss Tracking
- Qualitative Transcription Review

---

## 🛠️ Tech Stack

- Python
- PyTorch
- HuggingFace Transformers
- Torchaudio
- NumPy
- Pandas
- CUDA-enabled GPU

---

## 📂 Project Structure

├── data/
├── preprocessing/
├── training/
├── evaluation/
├── notebooks/
├── results/
└── README.md



---

## 🔍 Key Learnings

- Self-supervised learning significantly reduces dependency on large labeled datasets.
- Low-resource ASR systems are highly sensitive to hyperparameters.
- Dataset scaling directly impacts Word Error Rate performance.
- Training instability can be mitigated using:
  - Proper learning rate scheduling
  - Regularization
  - Gradient clipping

---

## 🎯 Future Work

- SpecAugment-based data augmentation
- Language model integration
- Multilingual fine-tuning
- Adapter-based fine-tuning
- Cross-lingual transfer learning

---

## 📌 Domain

- Automatic Speech Recognition (ASR)
- Self-Supervised Learning
- Low-Resource NLP
- Speech Processing
- Transformer-based Audio Modeling

---

## 📜 License

MIT License
