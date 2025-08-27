# Practical Application III — Comparing Classifiers (Bank Marketing)


## What this project is about

We compare four classifiers learned in this module — **Logistic Regression**, **K-Nearest Neighbors (KNN)**, **Decision Tree**, and **Support Vector Machine (SVM)** — on the **Bank Marketing** dataset (telephone campaigns by a Portuguese bank).

We:
- performed light **EDA** (class balance, distributions, categorical counts)  
- built **preprocessing pipelines** (impute + scale numeric, impute + one-hot categorical)  
- established a **baseline** (majority class)  
- trained/evaluated all four models with **default settings** and timed training  
- improved them with **GridSearchCV** (ROC AUC focus) and **optional threshold tuning** for better recall

**Primary metric:** **ROC AUC** (robust under class imbalance).  
We also report **Accuracy, Precision, Recall, F1** (positive class = “yes”).

---

## Business objective

Phone calls cost time and money. The bank wants to **contact customers who are most likely to subscribe to a term deposit**.  
Models should therefore **rank** and **identify** likely subscribers (higher **recall** with reasonable **precision**), not just maximize accuracy.

---

## Data notes

- Dataset: **Bank Marketing** (UCI ML Repository), 2008–2010 telephone campaigns  
- Target: `y` (“yes”/“no”) → mapped to **1/0**  
- Categorical unknowns are often labeled `'unknown'` → handled via imputation/one-hot  
- **Leakage:** `duration` is **excluded** (only known after the call)  
- **Special code:** `pdays == 999` = “not previously contacted” (optionally used as a helper feature)

---

## Environment & how to run

- Python 3.10+  
- `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`  
- Works on Google Colab (CPU fine; SVM is heavier)

**Steps**
1. Place the dataset CSV (e.g., `bank-full.csv` or `bank.csv`) next to the notebook.  
2. Open `prompt_III.ipynb`.  
3. Run all cells top to bottom.  
   - If the loader can’t find your file, edit the filename/`sep` in the data-loading cell.

---

## EDA highlights (short)

- Strong **class imbalance**: ~**88.7%** “no” vs **11.3%** “yes” in our split.  
- Numeric/categorical features as described in the UCI schema.  
- “Unknown” is a legitimate category in some columns; handled by the pipeline.

---

## Baseline

- **Majority class predictor** (always “no”) ≈ **88.7%** accuracy in our split  
- But: **Recall = 0** for “yes” and **ROC AUC = 0.5** (random).  
- Any useful model should **beat AUC 0.5** and improve recall for “yes”.

---

## Default model comparison (out-of-the-box)

| Model                | Train Time (s) | Train Accuracy | Test Accuracy |
|---------------------|----------------:|---------------:|--------------:|
| Logistic Regression | 0.231           | 0.8873         | **0.8874**    |
| KNN                 | 0.064           | 0.8910         | 0.8781        |
| Decision Tree       | 1.556           | 0.9171         | 0.8649        |
| SVM (RBF)           | 88.505          | 0.8882         | ~0.887        |

> Accuracy alone is misleading here because of imbalance; models with “high” accuracy can still miss most positives.

---

## Tuned models (GridSearchCV, ROC AUC focus)

We used small grids and stratified CV (3-fold for speed).  
**Your run’s results:**

| Model                      | Best CV ROC AUC | Train Time (s) | Test Accuracy | Test Precision | Test Recall | Test F1 | Test ROC AUC |
|---------------------------|----------------:|---------------:|--------------:|---------------:|------------:|--------:|-------------:|
| **Decision Tree (balanced)** | 0.685           | 11.77          | 0.8050        | 0.2713         | **0.4332**  | 0.3336  | **0.6985**   |
| Logistic Regression       | 0.696           | 46.61          | 0.8989        | 0.6753         | 0.1972      | 0.3053  | 0.6936       |
| KNN                       | 0.665           | 413.48         | 0.8973        | 0.6519         | 0.1897      | 0.2938  | 0.6711       |
| SVM (RBF)                 | 0.646           | 78.41          | 0.8988        | 0.6780         | 0.1929      | 0.3003  | 0.6198       |

**Takeaways**
- **Decision Tree (class_weight='balanced')** achieved the **best Test ROC AUC** and **highest recall**, catching more potential subscribers (key for the business goal).  
- **Logistic Regression** is a strong baseline with good AUC and very fast training; at the default threshold it predicts few positives (lower recall), but **lowering the threshold** increases recall.  
- **KNN** did not outperform Tree/LR and was slow in CV.  
- **SVM (RBF)** was computationally expensive and underperformed here.

---

## Threshold tuning (optional but useful)

After picking the “best” model by ROC AUC (Decision Tree in our run), we also explored **threshold tuning** using the **Precision–Recall curve**:
- **F1-optimal threshold** to balance precision/recall  
- **Target-recall threshold** (e.g., 0.45–0.60) to match call-center capacity or business priorities

> ROC AUC is threshold-independent; threshold tuning trades precision vs. recall for deployment.

---

## Recommendations

- If the goal is to **find more subscribers** (minimize false negatives), use the **Decision Tree (balanced)** and set an operating **threshold** to hit the desired recall.  
- If you want a simpler model with well-behaved probabilities, use **Logistic Regression** and **lower the decision threshold** to increase recall; monitor precision to control costs.  
- In production, score all clients, **rank by predicted probability**, and call the **top deciles** based on available agent capacity.  
- Track conversion rates by score decile and **retrain periodically** to prevent drift.

---

## Next steps

- Add more features (e.g., richer customer/product history)  
- Try **class weighting**, **oversampling** (SMOTE), or **calibrated probabilities**  
- Explore **ensemble** methods (Random Forest, Gradient Boosting)  
- **A/B test**: model-prioritized calls vs. current process; measure conversion and cost/call

---

## Repo Structure

```
.
├── prompt_III.ipynb           # main notebook
├── README.md                  # this file
└── data/
    └── bank-full.csv          # (optional) dataset if you keep it in-repo
```


