from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from common import load_split, NUM, CAT
def make_preprocess():
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", numeric, NUM), ("cat", categorical, CAT)])
if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = load_split()
    pre = make_preprocess()
    Xt = pre.fit_transform(X_train)
    print(Xt.shape)
    print(list(pre.get_feature_names_out()[[0, 4, 5, 10, 13]]))

# Output:
#   (3000, 22)
#   ['num__monthly_bill', 'num__age', 'cat__region_Central', 'cat__plan_Fiber 100', 'cat__tv_bundle_no']
# 5 numeric columns + 17 one-hot columns = 22.
# median: robust to outliers such as a 65-hour outage.
# handle_unknown="ignore": a new plan or region later will not crash predictions.
# The pipeline is fitted on TRAIN rows only and reuses the same medians, means and
# categories for every future row, so training and prediction always match.
