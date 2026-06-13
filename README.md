# Telecom Customer Churn Prediction (Srikakulam District Survey)

**Author:** Kranthi Kumar Palina
**Base data:** Independent 82-user telecom survey (2023-2026), Bavajipeta, Dusipeta, and Dusi villages, Srikakulam district, Andhra Pradesh

## Background

Over three years, I conducted a door-to-door survey of 82 mobile users (50 Jio, 28 Airtel, 4 BSNL) to understand real-world network issues in rural areas. The survey collected information on signal quality, billing complaints, plan-vs-actual speed mismatches, and switching behavior. Key findings included:

- 100% of Airtel users reported incoming calls being cut within 2 days of zero balance
- 100% of Jio users reported receiving 3 promotional calls daily after balance expiry
- 100% of all 82 users reported paying for 5G plans but experiencing 4G-level speeds at home
- ~60% of Airtel users reported having switched providers

## From Survey to Machine Learning

To extend this research into machine learning, I built a churn prediction model. Since individual-level data wasn't digitized during the original survey (only aggregate totals were recorded), I constructed a **synthetic dataset of 82 rows** that mirrors the exact percentages reported in the survey — for example, 40 of 50 Jio users (80%) were marked with the "PhonePe extra charge" complaint, matching the real survey finding.

For the churn label:
- **Airtel users**: ~60% were marked as "churned," based on the real switching rate reported in interviews
- **Jio/BSNL users**: churn was estimated using a complaint-count threshold (a proxy, since no real switching data was collected for these operators)

## Model and Results

I trained a **Logistic Regression** classifier (scikit-learn) using features such as operator, plan type, download speed, and seven binary complaint indicators (e.g., signal problems, post-expiry calls, extra recharge charges).

**Results on a held-out test set (21 users):**
- Accuracy: **95%**
- Precision/Recall for "Churned": 0.94 / 1.00
- Precision/Recall for "Stayed": 1.00 / 0.83

**Top predictors of churn (by model coefficient):**
1. PhonePe extra recharge charge (₹3) — by far the strongest predictor
2. Sitting under sunlight to get network signal
3. Age group 26-35
4. No 1GB/day plan available

## Key Insight

The most interesting finding was that **a small, repeated annoyance (the ₹3 PhonePe surcharge) was a stronger predictor of churn than the headline issue (5G plans delivering only 4G speeds)**. This suggests that for telecom providers, addressing minor recurring frustrations may have an outsized impact on customer retention compared to addressing larger infrastructure issues alone.

## Limitations and Honest Caveats

- The dataset is **synthetic**, constructed to match real aggregate survey statistics — not raw per-user records (which weren't digitized in the original study)
- Only the Airtel churn label is grounded in a real reported statistic (60% switching rate); Jio/BSNL labels are estimated proxies
- The high accuracy (95%) partly reflects the clean, pattern-based construction of the synthetic data; real individual-level data would likely show more noise and a lower accuracy
- BSNL sample size (4 users) is too small for reliable conclusions

## What's Next

- Follow up with original survey respondents to collect real "considering switching?" responses, replacing the proxy churn labels with ground-truth data
- Expand the feature set with recharge amount, exact monthly spend, and more granular location data
#NAME?

## Tools Used

Python, Pandas, NumPy, scikit-learn (Logistic Regression, OneHotEncoder, train/test split), Google Colab
<img width="91" height="1369" alt="image" src="https://github.com/user-attachments/assets/9beccccc-3f50-4f0a-93b1-9e9bfeb81412" />
