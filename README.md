# 📊 Marketing Campaign Analysis
### Customer Segmentation, Churn Prediction & Retention Intelligence

---

## 🎯 Business Problem

A marketing team is spending budget sending campaigns to **all customers equally**. This is wasteful and ineffective. This project answers four critical business questions:

- **Who are our best customers?**
- **Who is about to leave?**
- **Where should we spend our retention budget?**
- **Who is NOT worth spending money on?**

---

## 💡 Key Business Findings

| Metric | Result |
|--------|--------|
| Customer Segments Identified | 4 |
| Overall Churn Rate | ~25% |
| At-Risk Customers | 574 |
| Retention Campaign Cost | $8,610 |
| Estimated Revenue Saved | $22,752 |
| **ROI** | **164%** |

> Every $1 spent on targeted retention returns $2.60 in saved customer revenue.


## 🔄 Project Workflow

```
Raw Data (Excel)
      ↓
Data Cleaning & Feature Engineering
      ↓
Exploratory Data Analysis
      ↓
RFM Analysis (Recency, Frequency, Monetary)
      ↓
KMeans Clustering → Customer Segments
      ↓
Churn Definition & Prediction
      ↓
5 Model Comparison → Hyperparameter Tuning
      ↓
Feature Importance Analysis
      ↓
CLV Calculation + Business Recommendations + ROI
```

---

## 📊 Visualizations
<img width="1093" height="688" alt="MCA1" src="https://github.com/user-attachments/assets/5acbd342-5f46-4c0b-b33b-ced85cb1f9db" />
<img width="1101" height="780" alt="MCA2" src="https://github.com/user-attachments/assets/3d016af7-7733-48ac-9441-0dcb82688881" />
<img width="1266" height="432" alt="MCA3" src="https://github.com/user-attachments/assets/6749c9ac-dbd8-4398-b419-7499d35cb091" />
<img width="1117" height="777" alt="MCA4" src="https://github.com/user-attachments/assets/8d3ee4ab-3366-4092-a317-f7d9201e9956" />
<img width="1097" height="787" alt="MCA5" src="https://github.com/user-attachments/assets/e9d7786c-0fe1-4a8c-9dac-c1ca68be8ae0" />



## 🤖 Models Trained

| Model | Purpose | Key Metric |
|-------|---------|------------|
| Logistic Regression | Baseline interpretable model | Recall |
| SVM | Non-linear boundary detection | Recall |
| Random Forest | Ensemble, handles non-linear | ROC-AUC |
| **XGBoost** | **Best performer — selected** | **ROC-AUC** |
| LightGBM | Fastest, large dataset ready | ROC-AUC |

**Why Recall over Accuracy?**
Missing a churning customer (False Negative) costs far more than sending an unnecessary retention offer (False Positive). A lost Champion = hundreds in lost revenue. An unnecessary $15 coupon = $15 cost. We optimize to catch as many churners as possible.

**Why XGBoost over LR and SVM?**
XGBoost handles non-linear relationships automatically, provides native feature importance, and consistently outperforms linear models on real-world tabular data without requiring the scaling that SVM needs.

---

## 📈 Customer Lifetime Value

CLV calculated per segment using:

```
CLV = Avg Purchase Value × Total Purchases × (Tenure in Years)
```

Champions show significantly higher CLV justifying premium retention investment, while Lost customers show CLV often below the cost of a retention campaign.

---

## 💰 Retention ROI

```
Target Segment:     At-Risk Customers
Customers:          574
Campaign Cost:      $8,610  ($15 per customer)
Revenue Saved:      $22,752 (30% save rate assumed)
Net Gain:           $14,142
ROI:                164%
```

> Assumptions: $15 retention offer cost per customer, 30% of at-risk customers successfully retained. Adjust `retention_cost` and `save_rate` variables in Cell 22 for your actual campaign costs.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.x | Core language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Preprocessing, KMeans, LR, SVM, Random Forest |
| XGBoost | Best churn prediction model |
| LightGBM | Model comparison |
| Matplotlib / Seaborn | Visualizations |
| Jupyter Notebook | Development environment |


## 🔑 Key Learnings

**1. Feature engineering matters more than model choice**
Combining 6 spending columns into `MntTotalExpense` and 4 purchase channels into `TotalPurchases` made the model more interpretable and the business story cleaner.

**2. Target definition is a business decision not a technical one**
Churn at 180 days vs 90 days vs behavioral signals — each gives a different answer. We used a behavioral definition (low purchases + no campaign response + high browsing) because it captures intent to leave, not just absence.

**3. Interpretation beats accuracy**
A 164% ROI estimate communicates more to a marketing team than a 0.82 ROC-AUC score. Always translate model output into business language.

---

## 👩‍💻 Author
**Urvi Dhomne**
Washington DC | Open to Relocate
[LinkedIn](https://linkedin.com/in/urvidhomne) | [GitHub](https://github.com/urvidhomne)

---

*Dataset: Kaggle Marketing Campaign | Tools: Python, Scikit-learn, XGBoost | **Source:** [Kaggle — Marketing Campaign Dataset](https://www.kaggle.com/datasets/rodsaldanha/arketing-campaign/data)*
