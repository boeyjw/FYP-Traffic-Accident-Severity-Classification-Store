import matplotlib.pyplot as plt

from collections import OrderedDict
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from tap import modelmetrics

# Author: Kian Ho <hui.kian.ho@gmail.com>
#         Gilles Louppe <g.louppe@gmail.com>
#         Andreas Mueller <amueller@ais.uni-bonn.de>
#
# License: BSD 3 Clause

RANDOM_STATE = 123456789

# Generate a binary classification dataset.
print("Init")
X, y = joblib.load("stratified_X_train.pkl.z"), joblib.load("stratified_Y_train.pkl.z")
Xt, yt = joblib.load("stratified_X_test.pkl.z"), joblib.load("stratified_Y_test.pkl.z")
rfecv = joblib.load("rfecv_noCAS-res.pkl")

# NOTE: Setting the `warm_start` construction parameter to `True` disables
# support for parallelized ensembles but is necessary for tracking the OOB
# error trajectory during training.
ensemble_clfs = [
    ("max_features='sqrt'",
        RandomForestClassifier(warm_start=True, oob_score=True,
                                max_features="sqrt",
                                random_state=RANDOM_STATE, n_jobs=7)),
    ("max_features='log2'",
        RandomForestClassifier(warm_start=True, max_features='log2',
                                oob_score=True,
                                random_state=RANDOM_STATE, n_jobs=7)),
    ("max_features=None",
        RandomForestClassifier(warm_start=True, max_features=None,
                                oob_score=True,
                                random_state=RANDOM_STATE, n_jobs=7))
]
met = modelmetrics.metrics()

# Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)
all_met = OrderedDict((label, []) for label, _ in ensemble_clfs)

# Range of `n_estimators` values to explore.
min_estimators = 20
max_estimators = 110

print("OOB")
for label, clf in ensemble_clfs:
    for i in range(min_estimators, max_estimators + 1):
        clf.set_params(n_estimators=i)
        clf.fit(X[rfecv["after_sel_cols"]], y)

        # Record the OOB error for each `n_estimators=i` setting.
        oob_error = 1 - clf.oob_score_
        error_rate[label].append((i, oob_error))
        m = met.evaluate_model(y_true=yt, y_pred=clf.predict(Xt[rfecv["after_sel_cols"]]), name=label + ", n_estimators=" + str(len(clf.estimators_)))
        all_met[label].append((i, m))
        print(label + ": " + str(len(clf.estimators_)) + " - " + str(oob_error) + " - " + str(m['accuracy']))

print("Dump")
# Dump the oob score
joblib.dump(error_rate, "rf_oob-score_noCAS.pkl")
joblib.dump(all_met, "rf_oob-score-metrics_noCAS.pkl")

# Generate the "OOB error rate" vs. "n_estimators" plot.
for label, clf_err in error_rate.items():
    xs, ys = zip(*clf_err)
    plt.plot(xs, ys, label=label)

plt.xlim(min_estimators, max_estimators)
plt.xlabel("n_estimators")
plt.ylabel("OOB error rate")
plt.legend(loc="upper right")
plt.show()
