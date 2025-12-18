# DeepPhishDetect
Phishing URL Classification using Classical ML, Transformer Embeddings, and a Bidirectional LSTM.

##  Overview
DeepPhishDetect detects phishing URLs **using only the URL text**.  
We evaluate three modeling approaches:

1. Classical ML with handcrafted URL features  
2. Classical ML with transformer-based embeddings (CANINE-C)  
3. Bidirectional LSTM trained on raw character sequences  

The BiLSTM model achieves **~98% accuracy** and **0.92 F1** on in-distribution data, outperforming all classical methods.

---

##  Dataset
We combined multiple public sources to create a diverse dataset:

- **PhishTank** — ~46K recent phishing URLs  
- **Kaggle 500K dataset** — ~500K URLs (28% phishing)  
- **Independent Kaggle dataset** — used only as Test B for generalization

### Dataset Splits
| Split       | Phish    | Benign   | Ratio |
|-------------|----------|----------|-------|
| Train       | 147,137  | 147,137  | 1:1   |
| Validation  | 4,691    | 23,455   | 1:5   |
| Test A      | 9,382    | 93,820   | 1:10  |
| Test B      | 100,000  | 100,000  | 1:1   |

---

##  Models
### Classical ML (Handcrafted Features)
- Logistic Regression  
- SVM  
- Random Forest  
- XGBoost  

### Classical ML (Transformer Embeddings)
- CANINE-C embeddings → classical models  

### Deep Learning
- **Bidirectional LSTM**  
- Character-level tokenizer + embedding  
- Best overall performance  

---

## Results

### Test A (In-Distribution)
- **BiLSTM:** 98.5% accuracy, 86% precision, 99.8% recall, 0.92 F1  
- **Best classical (SVC):** 96.4% accuracy, 0.83 F1  

### Test B (Out-of-Distribution)
- Classical models collapsed (near-random)  
- **BiLSTM retained high phishing recall (~92%)** but lower benign precision  

---

## 🔍 Limitations
- URL-only models struggle under major distribution shifts  
- Real-world phishing detection requires:
  - WHOIS/domain age
  - Host/IP metadata
  - Page content features
  - Continuous retraining + drift monitoring

---

