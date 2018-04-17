import numpy as np
import pandas as pd

from scipy.stats import randint as sp_randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from tap import modelparams

# Reference: http://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html#sphx-glr-auto-examples-model-selection-plot-randomized-search-py
# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_f1_macro'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                results['mean_test_f1_macro'][candidate],
                results['std_test_f1_macro'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

if __name__ == "__main__":
    print("Init")
    params = modelparams.get_constants()
    rf = RandomForestClassifier(n_estimators=50, random_state=params["RANDOM_STATE"], oob_score=True, n_jobs=params["N_JOBS"])
    n_iter_torun = 20
    ext = ".cas.v2"
    param_dist = {
        "max_depth": [None] + list(range(3, 50, 2)),
        "min_samples_leaf": sp_randint(1, 100),
        "n_estimators": sp_randint(50, 150),
        "max_features": ["sqrt", "log2"]
    }
    rscv = RandomizedSearchCV(estimator=rf, n_iter=n_iter_torun, param_distributions=param_dist, scoring=["f1_macro", "precision_macro", "recall_macro", "accuracy"], refit="f1_macro", cv=StratifiedKFold(5), verbose=2, return_train_score=False)

    sample = pd.read_csv("train/stratified_XY_train.oh.tlsmote" + ext + ".csv")
    X = sample.drop("Accident_Severity", axis=1).copy()
    Y = sample["Accident_Severity"].copy()
    del sample

    print("Search RF")
    rscv.fit(X, Y)
    print("Dump")
    res = pd.DataFrame(rscv.cv_results_)
    res.to_csv("search/rf/randomsearch_rf" + ext + ".csv", index=False)
    report(rscv.cv_results_, n_top=n_iter_torun)
