***🎙️ Automatic Speech Recognition on Low Resource Languages using Self Supervised Languages.***
📌 Overview

This project focuses on building an Automatic Speech Recognition (ASR) system for low-resource Indian languages using state-of-the-art self-supervised speech models.

Low-resource languages suffer from limited labeled data, making traditional ASR approaches ineffective. To address this, this project leverages pretrained transformer-based speech models and fine-tunes them on limited labeled speech datasets.

🚀 Key Highlights

Fine-tuned Wav2Vec 2.0

Experimented with WavLM

Trained on Microsoft Indian Language Speech Corpus

Focused on Telugu and other low-resource Indian languages

Conducted dataset scaling experiments (small → medium → large subsets)

Analyzed training instability at intermediate steps (2000–2400 steps)

Evaluated performance using Word Error Rate (WER)

🧠 Problem Statement

Building robust ASR systems for low-resource languages is challenging due to:

Limited annotated speech data

Pronunciation variability

Dialectal diversity

Data imbalance

Overfitting on small datasets

This project investigates how self-supervised pretrained models can improve performance in such scenarios.

🏗️ Architecture

Pretrained Backbone

Wav2Vec 2.0 / WavLM (Transformer-based encoder)

Fine-tuning Strategy

CTC (Connectionist Temporal Classification) Loss

HuggingFace Transformers pipeline

Gradient accumulation for low GPU memory

Training Setup

Variable dataset sizes

Mixed precision training

Learning rate scheduling

Early stopping experiments

📊 Experiments
Dataset Scaling Experiments
Dataset Size	Observation
Small subset	Faster convergence but high overfitting
Medium subset	Improved generalization
Larger subset	Better WER but longer training time
Training Behavior Analysis

Observed:

Loss decreased steadily

At ~2000–2400 steps: temporary increase in validation loss

Possible causes:

Learning rate schedule spike

Overfitting

Gradient instability

Data distribution shift

This analysis helped optimize training configuration.

📈 Evaluation Metric

Word Error Rate (WER)

Validation loss tracking

Qualitative transcription analysis

🛠️ Tech Stack

Python

PyTorch

HuggingFace Transformers

Torchaudio

NumPy

Pandas

CUDA-enabled GPU training

📂 Project Structure
├── data/
├── preprocessing/
├── training/
├── evaluation/
├── notebooks/
├── results/
└── README.md
🔍 Key Learnings

Self-supervised models significantly reduce labeled data dependency.

Low-resource ASR is highly sensitive to hyperparameters.

Dataset scaling directly impacts WER performance.

Training instability can be mitigated using:

Proper LR scheduling

Regularization

Gradient clipping

🎯 Future Improvements

Data augmentation (SpecAugment)

Language Model integration

Multi-lingual fine-tuning

Adapter-based fine-tuning

Cross-lingual transfer learning

📌 Research & Domain

Domain:

Automatic Speech Recognition (ASR)

Self-Supervised Learning

Low-Resource NLP

Speech Processing

Transformer-based Audio Modeling

🤝 Contributions

Open to collaborations in:

Low-resource speech systems

Multilingual ASR

Speech foundation models

📜 License

MIT License
