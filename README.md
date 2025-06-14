# Detection of Power System Anomalies Using a Fusion of Machine Learning & Deep Learning

This repository contains the implementation and research artifacts for the paper:

**Detection of Power System Anomalies Using a Fusion of Machine Learning & Deep Learning**  
*By Jishnu Teja Dandamudi and Rupa Kandula*  
Amrita School of Artificial Intelligence, Amrita Vishwa Vidyapeetham, Coimbatore, Tamil Nadu, India

---

## 🧠 Overview

Anomaly detection in Phasor Measurement Unit (PMU) data is critical to maintaining stability and security in modern power grids. This project introduces a **hybrid anomaly detection framework** that combines:

- **Isolation Forest (IF)** for statistical anomaly detection
- **LSTM Autoencoder** for temporal anomaly detection

These techniques are fused using a weighted strategy to improve detection accuracy, particularly in the presence of missing data, noise, or cyber-induced anomalies.

---


---

## 🧪 Methodology

### LSTM Autoencoder
- Captures **temporal patterns** from sequential PMU data
- Uses **reconstruction error** to detect deviations from normal behavior

### Isolation Forest (IF)
- Identifies **statistical outliers** based on recursive partitioning
- Trained on the reconstruction errors from the LSTM

### Fusion Strategy
- Final anomaly score: **S_f(x) = α × S_LSTM(x) + β × S_IF(x)** where (α + β = 1)
- Classification is done using a **95th percentile threshold** on the final anomaly score


---

## 📊 Evaluation

Metrics used:
- **Precision:** 88.21%
- **Recall:** 88.29%
- **F1 Score:** 88.25%

Compared against state-of-the-art techniques like CyResGrid and GC-LSTM+ResNet.

---

## 📚 Dataset

Used the publicly available dataset:  
**Realistic Labelled PMU Data for Cyber-Power Anomaly Detection Using Real-Time Synchrophasor Testbed**  
Available on IEEE DataPort

---

## 🧩 Future Enhancements

- Adaptive thresholding using Bayesian Optimization or RL
- Self-supervised models using contrastive learning
- Federated learning for decentralized anomaly detection
- Integration of Explainable AI (XAI) for model interpretability

---
