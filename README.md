# CityNet churn baseline

End-to-end tabular ML baseline (AI Engineering Program, Module 3: SVM and the End-to-End Tabular ML Baseline, Labs 1-9).

## User story
As the retention manager, I want a weekly list of the 100 customers most likely to cancel next month, so my team can call them first.

## How to run
Run every command from the project folder (the folder with this README).

    python -m venv .venv
    .venv\Scripts\activate                        # Windows  (macOS/Linux: source .venv/bin/activate)
    pip install -r requirements.txt
    python src/lab7_save.py                       # train and save the pipeline
    python src/predict.py data/new_customers.csv  # writes reports/predictions.csv

Tested with Python 3.13, pandas 2.3.3, scikit-learn 1.7.2. Every random step uses seed 42, so reruns give the same numbers.

## Project layout

| Path | What it is |
|---|---|
| `data/` | `citynet_churn.csv` (5,000 customers, made up) and `new_customers.csv` (8 rows to score) |
| `DATA.md` | Data card: rows, target, missing values, the leaky column |
| `src/common.py` | Feature lists and the stratified 60/20/20 split used by every lab |
| `src/svm_toy.py` | Lab A: first linear SVM on six customers |
| `src/lab1_inspect.py` | Lab 1: inspect the data |
| `src/lab2_split.py` | Lab 2: check the split |
| `src/lab3_pipeline.py` | Lab 3: preprocessing pipeline (impute, scale, one-hot) |
| `src/lab4_models.py` | Lab 4: baseline, logistic regression, random forest (5-fold CV) |
| `src/lab4_leak_check.py` | Lab 4 cheat test: what happens with the leaky column |
| `src/lab4b_svm.py` | Lab 4b: linear SVM, with and without scaling, call list and weights |
| `src/lab5_evaluate.py` | Lab 5: validation metrics and the top-100 list |
| `src/lab6_errors.py` | Lab 6: importance, who we miss, fairness by region, test score |
| `src/lab7_save.py`, `src/predict.py` | Lab 7: save the pipeline, score new customers |
| `models/` | Saved pipeline (rebuilt by `lab7_save.py`, not kept in Git) |
| `reports/predictions.csv` | Scored `new_customers.csv`, highest risk first |

Each script ends with a comment block holding its real output and the answers to the lab questions.

## Results (validation, then test once)

| Metric | Baseline | Our model (logistic regression) |
|---|---|---|
| PR-AUC, 5-fold CV on training rows | 0.179 | 0.407 |
| PR-AUC, validation | 0.179 | 0.396 |
| PR-AUC, **test** (run once) | 0.179 | **0.361** |
| Churners in top 100, validation | about 18 (random) | 49 |
| Churners in top 100, test | about 18 (random) | 51 |

Model comparison (5-fold cross-validation on the 3,000 training rows):

| Model | PR-AUC | Recall | Precision |
|---|---|---|---|
| Baseline (always "stays") | 0.179 | 0.000 | 0.000 |
| Logistic regression | 0.407 | 0.669 | 0.306 |
| Random forest | 0.391 | 0.385 | 0.388 |
| Linear SVM (Lab 4b) | 0.406 | - | - |
| Linear SVM, no scaling | 0.394 | - | - |
| Logistic + leaky `cancel_request_logged` | 0.909 | - | - |

**Decision:** logistic regression goes forward. It has the best PR-AUC, ties with the linear SVM, gives probabilities, and is the simplest to explain. The 0.909 score with `cancel_request_logged` is leakage, not skill, so that column is never used.

**Why customers leave** (linear SVM weights, Lab 4b; permutation importance, Lab 6): short tenure, no auto-pay, paying the last bill late, and many support calls push towards leaving. Long tenure is the strongest reason to stay.

## Intended use
- Rank customers for a weekly retention call list: the retention team calls the 100 highest `churn_risk` customers.
- A human decides who to call and what to offer.

## Not for
- Changing prices or contracts automatically.
- Judging or penalising individual customers without a call.
- Any decision about customers outside the CityNet fibre base the model was trained on.

## Assumptions
- The last 90 days of behaviour predict cancellation next month.
- New customers look like the training data (same regions, plans and payment methods).
- The retention team can make about 100 calls a week, so we rank by risk instead of using a fixed 0.5 threshold.

## Risks and limitations
- **Misses quiet long-time customers:** churners the model misses have about 43 months of tenure on average, vs 19 months for those it catches.
- **Uneven across regions:** East churners are caught only 50% of the time, vs 68-76% in other regions. Check before relying on it in the East.
- **Many wrong calls at a 0.5 threshold:** precision is about 0.31, so roughly 2 of 3 flagged customers were not leaving. Ranking the top 100 is more precise (about half are real churners).
- **Snapshot data:** behaviour and prices drift. Retrain every quarter and re-check the top-100 hit rate.
- **Made-up data:** results show the method, not real CityNet customers. In a real company, customer data would never be committed to Git.

## Reproduce check (Lab 9)
A fresh `git clone` followed by the "How to run" commands gives the same results:
`python src/predict.py data/new_customers.csv` puts C90001 first (risk 0.797), and
`python src/lab5_evaluate.py` prints `top-100 churners: 49`. The regenerated
`reports/predictions.csv` is identical to the committed one. A different number means
a different split or library version: find out why before the demo.

## 5-minute demo
1. The user story and the six acceptance criteria
2. The data and the leaky column (`python src/lab1_inspect.py`, `python src/lab4_leak_check.py`)
3. The model table (`python src/lab4_models.py`, `python src/lab4b_svm.py`)
4. The top-100 result: 49 vs 18 on validation, 51 vs 18 on test
5. Who it misses: quiet long-time customers, and East region
6. Live: `python src/predict.py data/new_customers.csv`

## Acceptance criteria

- [x] 1. The top-100 list holds at least twice as many real churners as a random 100 (49 vs 18 on validation, 51 vs 18 on test)
- [x] 2. Test-set PR-AUC is reported (0.361) and beats the baseline (0.179)
- [x] 3. One command scores new customer rows (`python src/predict.py data/new_customers.csv`)
- [x] 4. No column that leaks the answer is used (`cancel_request_logged` excluded, see `DATA.md`)
- [x] 5. Data, assumptions, intended use and risks are written down (`DATA.md`, this README)
- [x] 6. Everything is in Git and reruns with seed 42
