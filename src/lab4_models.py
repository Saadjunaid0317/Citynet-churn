import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, precision_score
from common import load_split
from lab3_pipeline import make_preprocess
X_train, X_val, X_test, y_train, y_val, y_test = load_split()
candidates = {
    "baseline": DummyClassifier(strategy="most_frequent"),
    "logistic": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                     class_weight="balanced", random_state=42)}
rows = []
for name, model in candidates.items():
    pipe = Pipeline([("pre", make_preprocess()), ("model", model)])
    cv = cross_validate(pipe, X_train, y_train, cv=5, scoring={
        "pr_auc": "average_precision", "recall": "recall",
        "precision": make_scorer(precision_score, zero_division=0)})
    rows.append({"model": name, **{m: round(cv[f"test_{m}"].mean(), 3)
                                   for m in ["pr_auc", "recall", "precision"]}})
print(pd.DataFrame(rows).to_string(index=False))

# Output (5-fold CV on the 3,000 training rows, scikit-learn 1.7.2):
#      model  pr_auc  recall  precision
#   baseline   0.179   0.000      0.000
#   logistic   0.407   0.669      0.306
#     forest   0.391   0.385      0.388
# The baseline always predicts "stays": its PR-AUC equals the churn share (0.179) = no skill.
# Cheat test (src/lab4_leak_check.py): adding cancel_request_logged gives PR-AUC 0.909.
#   Far too good: that is leakage, so the column stays out.
# Decision: logistic regression goes forward. Best PR-AUC, and simpler to explain.
# (The forest's numbers differ slightly from the slides, likely a scikit-learn version
#  difference; the decision is the same.)
