import pandas as pd
df = pd.read_csv("data/citynet_churn.csv")
print(df.shape)
print(df.dtypes.value_counts().to_dict())
missing = df.isna().sum()
print(missing[missing > 0].to_dict())
print(df["churned"].value_counts(normalize=True).round(3).to_dict())
print(df[["tenure_months", "support_calls_90d", "outage_hours_90d"]].describe().loc[["mean", "min", "max"]].round(1))
print(df.groupby("support_calls_90d")["churned"].mean().round(2).head(6).to_dict())
print(df.groupby("cancel_request_logged")["churned"].mean().round(3).to_dict())

# TODO 1: Which column is missing values, and how many?
#   payment_method is missing 149 values and age is missing 411 values.
#   The Lab 3 pipeline fills them (most frequent value for payment_method, median for age).
# TODO 2: What share of customers churned? Is the data balanced?
#   17.9% churned (896 of 5000). No, it is imbalanced: a model that always says
#   "stays" is 82% accurate and finds zero churners, so we use PR-AUC and top-100 hits.
# TODO 3: One column predicts churn almost perfectly. Which one, and why must we NOT use it?
#   cancel_request_logged: 95.8% of customers with a logged cancel request churn, vs 3.3% without.
#   It is leakage: the request is logged AFTER the customer has decided to leave, so it is
#   not known when we build next month's call list. Using it gives great scores in the
#   notebook and a useless model in real life.
