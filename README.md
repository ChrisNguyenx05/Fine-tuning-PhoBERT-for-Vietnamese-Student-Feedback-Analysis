
 # Fine-tuning PhoBERT for Vietnamese Student Feedback Analysis

  [![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
  [![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
  [![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/)
  [![PhoBERT](https://img.shields.io/badge/Model-PhoBERT_Base-brightgreen.svg)](https://github.com/VinAIResearch/PhoBERT)


## Executive Summary
This project implements an end-to-end Natural Language Processing (NLP) pipeline to analyze and classify the sentiment of Vietnamese student feedback. By fine-tuning **PhoBERT** (a state-of-the-art language model for Vietnamese), the system automatically categorizes student reviews regarding facilities, lecturers, and curriculum into three distinct polarities: **Positive, Neutral, and Negative**. 

This solution demonstrates strong competencies in **Large Language Model (LLM) fine-tuning, sequence classification, data preprocessing (Vietnamese Word Segmentation), and model evaluation**.

## Key Features
- **State-of-the-Art Language Representation:** Leverages `vinai/phobert-base` to capture the deep contextual semantics of the Vietnamese language.
- **Robust Text Preprocessing:** Utilizes `ViTokenizer` for optimal Vietnamese word segmentation prior to tokenization.
- **High-Performance Inference:** Capable of classifying raw textual feedback in real-time with exceptional accuracy.
- **Scalable Architecture:** Built on top of `PyTorch` and `Hugging Face Transformers`, making it highly adaptable for production deployment.

## Technical Architecture & Stack
- **Core Language:** Python
- **Deep Learning Framework:** PyTorch
- **NLP Ecosystem:** Hugging Face `transformers` & `datasets`
- **Vietnamese Tokenization:** `pyvi` (ViTokenizer)
- **Metrics & Evaluation:** `scikit-learn`
- **Hardware Acceleration:** NVIDIA L4 GPU (Google Colab)

## Model Performance & Results
The model was fine-tuned over 3 epochs and rigorously evaluated on a test set of **3,166** unseen samples, achieving outstanding metrics:

- **Overall Accuracy:** **94.6%**
- **Weighted F1-Score:** **94%**

**Detailed Classification Report:**
| Class | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: |
| **Tiêu cực (Negative)** | 0.94 | 0.97 | 0.96 |
| **Trung tính (Neutral)** | 0.75 | 0.55 | 0.63 |
| **Tích cực (Positive)** | 0.95 | 0.96 | 0.95 |

## Quick Start & Inference
To run the model locally on a piece of text:

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pyvi import ViTokenizer

# 1. Load Model & Tokenizer
model_path = "./phobert_sentiment_final"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# 2. Preprocess Input
text = "Thầy nhiệt tình, slide đẹp, rất thích môn này."
text_segmented = ViTokenizer.tokenize(text)
inputs = tokenizer(text_segmented, return_tensors="pt", max_length=128, padding='max_length', truncation=True)

# 3. Predict
outputs = model(**inputs)
prediction = torch.argmax(outputs.logits, dim=1).item()
classes = ['Tiêu cực 😡', 'Trung tính 😐', 'Tích cực 😃']

print(f"Prediction: {classes[prediction]}")
# Output: Prediction: Tích cực 😃
```

## Contact & Connect
Designed and developed as a Personal Project highlighting advanced Data Science and NLP engineering capabilities. 
Feel free to reach out if you'd like to discuss the methodology or potential applications!
