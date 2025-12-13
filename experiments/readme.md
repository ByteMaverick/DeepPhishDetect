# Experimental Setup for Phishing URL Detection

This document outlines the three major dataset configurations and the testing pipeline used to train and evaluate machine-learning models for predicting whether a URL is phishing or legitimate.

---

## 1. URL-Based Feature Dataset (Basic Feature Set)

This dataset contains **19 handcrafted features**, primarily derived directly from the URL string.  
Included are **12 core lexical and structural features**, such as:

- URL length  
- URL entropy  
- Presence of an IP address  
- Presence of suspicious keywords  
- Special character frequency  
- Subdomain depth  
- Other direct URL-derived characteristics  

This serves as the **baseline dataset** focusing on lightweight, URL-only feature extraction.

---

## 2. Enhanced Feature Engineering Dataset (Expanded Feature Set)

This dataset extends the basic feature set by incorporating **additional behavior- and pattern-oriented features**, including:

- Suspicious or uncommon top-level domain (TLD) detection  
- Embedded or encoded character detection  
- Flags for excessively long URLs  
- Extended suspicious keyword and pattern checks  
- More detailed structural irregularity indicators  

This represents a **richer and more expressive engineered dataset** to potentially improve classical ML model performance.

---

## 3. Transformer-Based URL Embedding Dataset (DistilWord Embedding Model)

This dataset uses a pretrained transformer model (**DistilWord**) to generate **dense vector embeddings** from URLs.

Key elements include:

- Tokenizing URLs for transformer input  
- Producing high-dimensional embeddings  
- Training ML models or neural classifiers directly on these embeddings  

This reflects a **representation-learning approach** rather than manual feature engineering.

---

## 4. Unified Testing & Evaluation Pipeline

All three dataset configurations will be evaluated using a **consistent experimental pipeline**:

- Standard ML models (e.g., Logistic Regression, Random Forest, XGBoost, etc.)  
- Train/validation/test splits  
- Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC  
- Direct performance comparisons across dataset types  

---
