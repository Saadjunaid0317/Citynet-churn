from common import load_split
X_train, X_val, X_test, y_train, y_val, y_test = load_split()
print(len(X_train), len(X_val), len(X_test))
print(round(y_train.mean(), 3), round(y_val.mean(), 3), round(y_test.mean(), 3))

# Output:
#   3000 1000 1000
#   0.179 0.179 0.179
# 60 / 20 / 20 split. stratify keeps the same 17.9% churn share in every part.
# Every later lab imports load_split(), so the split is identical everywhere.
# Train: learn. Validation: choose. Test: report, once, at the end.
