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

---

## 📁 Project Structure

```
marketing-campaign-analysis/
│
├── marketing_campaign_jupyter.py   # Full analysis notebook
├── marketing_campaign.xlsx         # Raw dataset (Kaggle)
├── customer_segments_results.csv   # Model output — segment labels + churn scores
│
├── images/
│   ├── eda_overview.png            # Customer behavior distributions
│   ├── correlation_heatmap.png     # Feature correlations
│   ├── elbow_plot.png              # Optimal cluster selection
│   ├── segment_profiles.png        # RFM profiles per segment
│   └── feature_importance.png      # Churn drivers
│
└── README.md
```

---

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

### Customer Behavior Overview
![EDA Overview](images/eda_overview.png)

*Distribution of spending, age, purchases, and campaign response rate across 2,216 customers.*

---

### Feature Correlations
![Correlation Heatmap](images/correlation_heatmap.png)

*Income and total spend show the strongest positive correlation with campaign response.*

---

### Optimal Number of Segments
![Elbow Plot](images/elbow_plot.png)

*Elbow method and silhouette scores used to justify k=4 clusters. Picking k arbitrarily without this analysis risks creating meaningless segments.*

---

### Customer Segment Profiles
![Segment Profiles](images/segment_profiles.png)

*Four distinct segments identified by RFM behavior. Champions spend 10x more than Lost customers and respond to campaigns at significantly higher rates.*

---

### Churn Drivers
![Feature Importance](images/feature_importance.png)

*Top drivers of churn:*
- **TotalPurchases** — infrequent buyers churn fastest
- **NumWebVisitsMonth** — high browsing with no buying = disengaging customer
- **AcceptedCmp_Total** — customers who never respond to any campaign are checked out

---

## 👥 Customer Segments

| Segment | Behavior Profile | Action |
|---------|-----------------|--------|
| 🏆 **Champions** | Low recency, high frequency, high spend | Reward and protect — VIP access, referral requests |
| ⭐ **Loyal Customers** | Medium everything, consistent buyers | Upsell and cross-sell — personalized recommendations |
| ⚠️ **At-Risk** | Rising recency, dropping engagement | **Urgent** — time-limited win-back offer, multichannel |
| 📉 **Lost Customers** | High recency, low frequency, low spend | Minimal investment — one reactivation email only |

---

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

---

## 🚀 How to Run

**1. Clone the repository**
```bash
git clone https://github.com/urvidhomne/marketing-campaign-analysis.git
cd marketing-campaign-analysis
```

**2. Install dependencies**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm openpyxl jupyter
```

**3. Add the dataset**

Download `marketing_campaign.xlsx` from [Kaggle](https://www.kaggle.com/datasets/rodsaldanha/arketing-campaign/data) and place it in the root folder.

**4. Run the notebook**
```bash
jupyter notebook marketing_campaign_jupyter.py
```

Run cells top to bottom. Each cell has Q&A comments explaining what and why.

**5. Find your outputs**
```
eda_overview.png
correlation_heatmap.png
elbow_plot.png
segment_profiles.png
feature_importance.png
customer_segments_results.csv   ← segment labels + churn probabilities for every customer
```

---

## 📋 Dataset

**Source:** [Kaggle — Marketing Campaign Dataset](https://www.kaggle.com/datasets/rodsaldanha/arketing-campaign/data)

**Size:** 2,240 customers × 29 features

**Key columns used:**
- `Recency` — days since last purchase
- `MntWines`, `MntFruits`, etc. → combined into `MntTotalExpense`
- `NumWebPurchases`, `NumStorePurchases`, etc. → combined into `TotalPurchases`
- `AcceptedCmp1-5` → combined into `AcceptedCmp_Total`
- `Response` — response to last campaign (original target)
- `IsChurned` — engineered churn label (our primary target)

---

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

*Dataset: Kaggle Marketing Campaign | Tools: Python, Scikit-learn, XGBoost*
