import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix
from common import load_split
from lab3_pipeline import make_preprocess
def train_model(X_train, y_train):
    pipe = Pipeline([("pre", make_preprocess()),
                     ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))])
    return pipe.fit(X_train, y_train)
if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = load_split()
    pipe = train_model(X_train, y_train)
    proba = pipe.predict_proba(X_val)[:, 1]
    print("PR-AUC:", round(average_precision_score(y_val, proba), 3))
    print(confusion_matrix(y_val, proba >= 0.5).tolist())
    print(classification_report(y_val, proba >= 0.5, digits=3))
    top = np.argsort(-proba)[:100]                # the call team's 100 calls
    hits = y_val.to_numpy()[top].sum()
    print("top-100 churners:", hits, "| random 100 would find about", round(100 * y_val.mean()))

# Output (validation set, 1,000 customers):
#   PR-AUC: 0.396
#   [[541, 280], [56, 123]]        rows: real stays, real churners; cols: predicted stays, churners
#   churners: precision 0.305, recall 0.687
#   top-100 churners: 49 | random 100 would find about 18
# Acceptance criterion 1: 49 vs 18 is about 2.7x a random list, so it passes (needs >= 2x).
# In business terms: "Call our top 100 and about half were planning to leave.
#   100 random calls find about 18."
#
# Your turn, threshold 0.5 -> 0.3:
#   0.5: [[541, 280], [56, 123]]  precision 0.305  recall 0.687  (403 customers flagged)
#   0.3: [[294, 527], [18, 161]]  precision 0.234  recall 0.899  (688 customers flagged)
#   More churners caught (161 vs 123) but many more wrong calls (527 vs 280).
#   The right threshold depends on how many calls the team can make; with 100 calls
#   a week we simply rank by risk and take the top 100.
