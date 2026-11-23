# Lab 4, "your turn": the cheat test.  Run: python src/lab4_leak_check.py
# Same logistic model as Lab 4, but with the leaky column cancel_request_logged added.
# common.py is NOT changed, so every other lab stays leak-free.
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from common import NUM, CAT

LEAKY_NUM = NUM + ["cancel_request_logged"]

df = pd.read_csv("data/citynet_churn.csv")
X, y = df[LEAKY_NUM + CAT], df["churned"]
# same first split as common.load_split(), so these are the same 3,000 training rows
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)

pre = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                      ("scale", StandardScaler())]), LEAKY_NUM),
    ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CAT)])
pipe = Pipeline([("pre", pre),
                 ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))])
cv = cross_validate(pipe, X_train, y_train, cv=5, scoring="average_precision")
print("logistic WITH cancel_request_logged, PR-AUC:", round(cv["test_score"].mean(), 3))
