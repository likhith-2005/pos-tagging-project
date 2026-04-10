# POS Tagging Project

## 📌 Overview
This project implements multiple Part-of-Speech (POS) tagging models and compares their performance.

## 🚀 Models Implemented
- Rule-Based Model
- Hidden Markov Model (HMM)
- BiLSTM Model
- Conditional Random Field (CRF)
- Context-Based Model
- Custom Tagset Model

## 📊 Results

| Model | Accuracy |
|------|---------|
| Rule-Based | 0.0347 |
| HMM | 0.8678 |
| BiLSTM | 0.9592 |
| CRF | 0.9858 |
| Custom Tagset | 0.1096 |

## 📈 Analysis
- CRF achieved the highest accuracy
- BiLSTM also performed very well
- Rule-based and custom models performed poorly due to lack of learning

## ⚙️ How to Run

### ▶ Run Basic Analysis
```bash
python src/analysis.py

python src/analysis_advanced.py

python src/graph_analysis.py

python src/crf_model.py

python src/context_based_tagger.py