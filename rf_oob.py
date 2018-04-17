import matplotlib.pyplot as plt

from multiprocessing import cpu_count
from collections import OrderedDict
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from tap import modelmetrics, modelparams

# Author: Kian Ho <hui.kian.ho@gmail.com>
#         Gilles Louppe <g.louppe@gmail.com>
#         Andreas Mueller <amueller@ais.uni-bonn.de>
#
# License: BSD 3 Clause
# Reference: http://scikit-learn.org/stable/auto_examples/ensemble/plot_ensemble_oob.html

params = modelparams.get_constants()
RANDOM_STATE = params['RANDOM_STATE']
N_JOBS = params['N_JOBS'] # Leave 1 thread for system use (extremely important during thrashing)

print("Init")
# Import training dataset
# X, y = joblib.load("stratified_X_train.pkl.z"), joblib.load("stratified_Y_train.pkl.z")
ext = ".nocas.v2"
sample = joblib.load("train/stratified_XY_train.oh.tlsmote" + ext + ".pkl.xz")
# Split out validation set
X_train, X_test, Y_train, Y_test = train_test_split(sample["X2"], sample["Y"], test_size=params["TEST_SIZE"], random_state=RANDOM_STATE, stratify=sample["Y"])
# del X, y # Save some memory as original data has been split
del sample # Save some memory as original data has been split

# Load selected features
# rfecv = joblib.load("rfecv_withCAS-res.pkl")
# rfecv = {
#     "after_sel_cols": joblib.load("rfecv_withCAS-res.pkl")["after_sel_cols"].tolist() + joblib.load("rfecv_noCAS-res.pkl")["after_sel_cols"].tolist()
# }

# NOTE: Setting the `warm_start` construction parameter to `True` disables
# support for parallelized ensembles but is necessary for tracking the OOB
# error trajectory during training.
ensemble_clfs = [
    ("max_features=sqrt",
        RandomForestClassifier(warm_start=True, oob_score=True,
                                max_features="sqrt",
                                random_state=RANDOM_STATE, n_jobs=N_JOBS)),
    ("max_features=log2",
        RandomForestClassifier(warm_start=True, max_features="log2",
                                oob_score=True,
                                random_state=RANDOM_STATE, n_jobs=N_JOBS)),
    ("max_features=all",
        RandomForestClassifier(warm_start=True, max_features=None,
                                oob_score=True,
                                random_state=RANDOM_STATE, n_jobs=N_JOBS))
]
met = modelmetrics.metrics()

# Map max_features to a list of (<n_estimators>, <error rate>) pairs.
error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)
# Map max_features name to a list of (<n_estimators>, <metrics>) pairs.
all_met = OrderedDict((label, []) for label, _ in ensemble_clfs)

# Range of `n_estimators` values to explore.
min_estimators = 50 # Any less will cause trees to learn insufficient features
max_estimators = 150 # Any more wil take ages to complete due to thrashing

print("OOB")
for label, clf in ensemble_clfs:
    for i in range(min_estimators, max_estimators + 1):
        clf.set_params(n_estimators=i)
        clf.fit(X_train, Y_train)

        # Record the OOB error and validation metrics for each `n_estimators=i` setting.
        oob_error = 1 - clf.oob_score_
        error_rate[label].append((i, oob_error))
        m = met.evaluate_model(y_true=Y_test, y_pred=clf.predict(X_test), name=label + ", n_estimators=" + str(len(clf.estimators_)))
        all_met[label].append((i, m))
        print(label + ": " + str(len(clf.estimators_)) + " - " + str(oob_error) + " - " + str(m['f1_macro']))

del clf
print("Dump: " + ext)
# Dump the oob score
joblib.dump(error_rate, "rf_oob/rf_oob-score" + ext + ".pkl.xz")
joblib.dump(all_met, "rf_oob/rf_oob-score-metrics" + ext + ".pkl.xz")

# Generate the "OOB error rate" vs. "n_estimators" plot.
for label, clf_err in error_rate.items():
    xs, ys = zip(*clf_err)
    plt.plot(xs, ys, label=label)

plt.xlim(min_estimators, max_estimators)
plt.xlabel("Number of Estimators")
plt.ylabel("OOB Error Rate")
plt.title("OOB Error Rate against Number of Estimators for Casualty Inclusive Features")
plt.legend(loc="upper right")
plt.show()
