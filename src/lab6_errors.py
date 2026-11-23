import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score
from common import load_split
from lab5_evaluate import train_model
X_train, X_val, X_test, y_train, y_val, y_test = load_split()
pipe = train_model(X_train, y_train)
imp = permutation_importance(pipe, X_val, y_val, scoring="average_precision",
                             n_repeats=10, random_state=42)
print(pd.Series(imp.importances_mean, index=X_val.columns).nlargest(4).round(3).to_dict())
val = X_val.assign(churned=y_val.to_numpy(), caught=pipe.predict(X_val) == 1)
churners = val[val.churned == 1]
print(churners.groupby("caught")[["tenure_months", "outage_hours_90d"]].mean().round(1))
print(churners.groupby("region")["caught"].mean().round(2).to_dict())
print("TEST PR-AUC, run once at the end:",
      round(average_precision_score(y_test, pipe.predict_proba(X_test)[:, 1]), 3))

# Output:
#   importance: {'tenure_months': 0.13, 'auto_pay': 0.044, 'support_calls_90d': 0.043, 'paid_late_last_bill': 0.026}
#   churners missed (caught=False): tenure 43.2 months, outage 6.1 h
#   churners caught (caught=True):  tenure 18.9 months, outage 8.6 h
#   caught by region: Central 0.76, North 0.73, West 0.71, South 0.68, East 0.50
#   TEST PR-AUC, run once at the end: 0.361
#
# Findings:
# - Most important: tenure, then auto-pay, support calls and paying late.
# - Who does the model miss? Long-time customers (about 43 months) who leave quietly,
#   without many outages or calls.
# - Is it fair across regions? No: East churners are caught only half the time (0.50)
#   vs 0.68-0.76 elsewhere.
# - Honest score: test PR-AUC 0.361 (validation 0.396). Baseline is 0.179, so it beats it.
#   On the test set the top 100 holds 51 churners vs about 18 at random (2.8x).
# - The test set was used here once. No tuning after this point.
