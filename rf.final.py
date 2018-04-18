import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from tap import modelmetrics, modelparams

print("Init")
params = modelparams.get_constants()
met = modelmetrics.metrics()
test = joblib.load("test/stratified_traintest.oh.pkl.xz")
rf = RandomForestClassifier(n_jobs=params["N_JOBS"], oob_score=True, min_samples_leaf=1, max_features="log2")

print("nocas")
# nocas
train = joblib.load("train/stratified_XY_train.oh.tlsmote.nocas.v2.pkl.xz")
rf.set_params(max_depth=45, n_estimators=80)
y_pred = rf.fit(train["X2"], train["Y"]).predict(test["x"][train["X2"].columns])
res = met.evaluate_model(y_true=test["y"], y_pred=y_pred, name="nocas-final", mode="macro", store_y_pred=True)
joblib.dump({
    "model": rf,
    "result": res
}, "final/rf.final.nocas.v2.pkl.xz")

print("cas")
# cas
train = joblib.load("train/stratified_XY_train.oh.tlsmote.cas.v2.pkl.xz")
rf.set_params(max_depth=43, n_estimators=100)
y_pred = rf.fit(train["X2"], train["Y"]).predict(test["x"][train["X2"].columns])
res = met.evaluate_model(y_true=test["y"], y_pred=y_pred, name="nocas-final", mode="macro", store_y_pred=True)
joblib.dump({
    "model": rf,
    "result": res
}, "final/rf.final.cas.v2.pkl.xz")
