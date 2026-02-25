# 📊 Marketing Campaign Analysis
### Customer Segmentation, Churn Prediction & Retention Intelligence

---

## 🎯 Business Problem

A marketing team is spending budget sending campaigns to **all customers equally**. This is wasteful and ineffective. This project answers four critical business questions:

**Who are our best customers?**
→ Champions segment — 541 customers, avg CLV $1,248, 0% churn rate. Identified through KMeans clustering on RFM scores. 

**Who is about to leave?**
→ At-Risk segment — 574 customers, 27% churn rate. Predicted using LightGBM churn model with 60.3% recall. 

**Where should we spend our retention budget?**
→ At-Risk customers. $15 retention offer per customer yields 164% ROI. Champions need VIP treatment, not discounts. 

**Who is NOT worth spending money on?**
→ Lost Customers — 606 customers, avg CLV $131. Retention cost exceeds their lifetime value. One reactivation email only, then remove from active marketing list.

---
## 💡 Key Business Findings
<img width="615" height="162" alt="image" src="https://github.com/user-attachments/assets/f36d9e5c-bb88-4bf7-9dfe-862c8632dd3d" />

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

---
## 💰 Retention ROI

<img width="543" height="291" alt="image" src="https://github.com/user-attachments/assets/c7161bc8-1f2d-403f-970d-2c9061da4219" />

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

| Top Churn Driver | TotalPurchases (56%) |
| 2nd Driver | Web visits without buying (34%) |
| 3rd Driver | Zero campaign response (9%) |


## 👩‍💻 Author: Urvi Dhomne
---
*Dataset: Kaggle Marketing Campaign | Tools: Python, Scikit-learn, XGBoost | **Source:** [Kaggle — Marketing Campaign Dataset](https://www.kaggle.com/datasets/rodsaldanha/arketing-campaign/data)*
