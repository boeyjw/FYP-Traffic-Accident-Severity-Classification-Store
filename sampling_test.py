from collections import OrderedDict
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from tap import modelmetrics, modelparams

p = modelparams.get_constants()
met = modelmetrics.metrics()
cattap = joblib.load("stratified_traintest.oh.pkl.xz")
# scores = OrderedDict([
#     (s, -1) for s in ["tl", "enn","smote","smoteenn","smotetl"]
# ])
scores = OrderedDict([
    (s, -1) for s in ["smoteenn","smotetl"]
])

rf = RandomForestClassifier(n_estimators=50, n_jobs=p["N_JOBS"], random_state=p["RANDOM_STATE"])

for k, _ in scores.items():
    print(k)
    sampled = joblib.load("stratified_XY_train.oh." + k + ".pkl.xz")
    scores[k] = met.evaluate_model(y_true=cattap["y"], y_pred=rf.fit(sampled["X"], sampled["Y"]).predict(cattap["x"].drop(['Accident_Index', 'Date_Time', 'Year'], axis=1)), name=k, do_print=True)

joblib.dump(scores, "sampler_scores_combine.pkl.xz")
