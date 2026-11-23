# Lab A: your first SVM on 6 customers.  Run: python src/svm_toy.py
from sklearn.svm import SVC

X = [[60, 0], [48, 1], [40, 1],      # tenure, calls: stayed
     [5, 6],  [10, 4], [14, 5]]      # left
y = [0, 0, 0, 1, 1, 1]

model = SVC(kernel="linear").fit(X, y)

print("support vectors:", model.support_vectors_.tolist())
print("new customers:", model.predict([[50, 1], [8, 5], [30, 2]]).tolist())
print("scores:", model.decision_function([[50, 1], [8, 5], [30, 2]]).round(2).tolist())

# Output:
#   support vectors: [[40.0, 1.0], [14.0, 5.0]]
#   new customers: [0, 1, 0]
#   scores: [-1.75, 1.45, -0.24]
# A (50 months, 1 call) stays, B (8 months, 5 calls) leaves, C (30 months, 2 calls) stays.
# C's score is only -0.24, so it sits inside the street: a weak "stays".
# That is why we rank customers by the score instead of using only 0 / 1.
