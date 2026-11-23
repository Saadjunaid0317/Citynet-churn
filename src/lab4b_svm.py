# Lab 4b: a linear SVM on CityNet.  Run: python src/lab4b_svm.py
import time
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from common import load_split, NUM, CAT
from lab3_pipeline import make_preprocess

X_train, X_val, X_test, y_train, y_val, y_test = load_split()

def no_scaling():
    # Lab 3 preprocessing with the StandardScaler removed
    return ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUM),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CAT)])

def test(name, prep, model):
    pipe = Pipeline([("prep", prep), ("model", model)])
    start = time.time()
    cv = cross_validate(pipe, X_train, y_train, cv=5, scoring="average_precision")
    print(f"{name:24s} PR-AUC: {cv['test_score'].mean():.3f}   time: {time.time() - start:.0f}s")

# Part 1: compare
# TODO before running: guess which line takes the longest, and why
#   The "no scaling" line. Without StandardScaler, monthly_bill (thousands) dwarfs the
#   0/1 one-hot columns, so the optimiser needs many more steps to find the widest street.
test("SVM linear", make_preprocess(), SVC(kernel="linear", class_weight="balanced"))
test("SVM linear, no scaling", no_scaling(), SVC(kernel="linear", class_weight="balanced"))

# Part 2: use it like Lab 5
pipe = Pipeline([("prep", make_preprocess()), ("model", SVC(kernel="linear", class_weight="balanced"))])
pipe.fit(X_train, y_train)
score = pipe.decision_function(X_val)
top = np.argsort(-score)[:100]
print("top-100 churners:", y_val.to_numpy()[top].sum(), "| random 100 would find about", round(100 * y_val.mean()))

# Part 3: which columns push a customer towards leaving?
w = pd.Series(pipe[-1].coef_[0], index=pipe[0].get_feature_names_out())
print("push to LEAVE:", w.nlargest(3).round(2).to_dict())
print("push to STAY: ", w.nsmallest(3).round(2).to_dict())

# Output (scikit-learn 1.7.2):
#   SVM linear               PR-AUC: 0.406   time: 1s
#   SVM linear, no scaling   PR-AUC: 0.394   time: 64s
#   top-100 churners: 51 | random 100 would find about 18
#   push to LEAVE: {'cat__paid_late_last_bill_yes': 0.35, 'cat__auto_pay_no': 0.26, 'num__support_calls_90d': 0.25}
#   push to STAY:  {'num__tenure_months': -0.77, 'cat__paid_late_last_bill_no': -0.35, 'cat__auto_pay_yes': -0.26}
#
# Ticket SVM-101 answers:
# 1. PR-AUC: linear SVM 0.406 vs logistic 0.407 -> a tie. Without scaling it is worse
#    (0.394) and about 60x slower, so always scale inside the pipeline.
# 2. The SVM's top 100 holds 51 real churners vs about 18 for a random 100 (2.8x).
# 3. Reasons to LEAVE: paid the last bill late, no auto-pay, many support calls.
#    Reasons to STAY: long tenure (strongest), paid on time, auto-pay on.
# Same score as logistic -> ship logistic (probabilities, simpler to explain).
