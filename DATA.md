# Data card: citynet_churn.csv

- Source: CityNet customer records (made-up data for this course)
- Rows: 5,000   Columns: 14 (7 text, 4 integer, 3 decimal)
- One row is: one customer, measured over the last 90 days
- Target: `churned` = 1 if the customer cancelled in the next month. Share of 1s: 0.179 (896 of 5,000), so the classes are imbalanced
- Missing values: `payment_method` 149 rows, `age` 411 rows. No other column has gaps; no duplicate rows. Gaps are filled inside the pipeline using training rows only (most frequent value / median)
- Columns NOT used, and why:
  - `customer_id`: an identifier, carries no information about behaviour
  - `cancel_request_logged`: **leakage**. It is logged after the customer has already decided to leave (churn rate 95.8% when set vs 3.3% when not), so it is not available when the weekly call list is made. Adding it pushes PR-AUC to about 0.9 on paper and makes the model useless in practice
- Features used (11): numeric `monthly_bill`, `tenure_months`, `support_calls_90d`, `outage_hours_90d`, `age`; categorical `region` (5 values), `plan` (Fiber 20/50/100), `tv_bundle`, `payment_method` (bank transfer, card, cash), `auto_pay`, `paid_late_last_bill`
- Known issues:
  - Imbalanced target: accuracy is misleading, so PR-AUC and churners in the top 100 are used
  - `outage_hours_90d` has outliers (max 65.9 h vs mean 7.2 h), so numeric gaps are filled with the median
  - One snapshot in time; customer behaviour and prices can drift
  - Synthetic data: results show the method, not real CityNet behaviour
