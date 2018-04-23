from collections import OrderedDict, Counter
from sklearn.externals import joblib
from tap import modelparams
from imblearn.under_sampling import TomekLinks, EditedNearestNeighbours
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN, SMOTETomek

p = modelparams.get_constants()

## Oversample first then undersample
smote = SMOTE(random_state=p["RANDOM_STATE"], n_jobs=p["N_JOBS"])
undersampler = OrderedDict([
    ("smoteenn", EditedNearestNeighbours(random_state=p["RANDOM_STATE"], n_jobs=p["N_JOBS"], ratio="all")),
    ("smotetl", TomekLinks(random_state=p["RANDOM_STATE"], n_jobs=p["N_JOBS"], ratio="all"))
])
smotetap = joblib.load("stratified_XY_train.oh.smote.pkl.xz")

for k, v in undersampler.items():
    print("Start: " + k)
    newX, newY = v.fit_sample(smotetap["X"], smotetap["Y"]) # Data cleaning
    joblib.dump({"X": newX, "Y": newY}, "stratified_XY_train.oh." + k + ".pkl.xz")
    print("Count: " + str(Counter(newY)))

