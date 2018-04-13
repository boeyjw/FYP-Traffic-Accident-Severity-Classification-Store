from collections import OrderedDict, Counter
from sklearn.externals import joblib
from tap import modelparams
from imblearn.under_sampling import TomekLinks, EditedNearestNeighbours
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN, SMOTETomek

p = modelparams.get_constants()

sampler = OrderedDict([
    ("tl", TomekLinks(random_state=p["RANDOM_STATE"], n_jobs=p["N_JOBS"], ratio="majority")),
    ("enn", EditedNearestNeighbours(random_state=p["RANDOM_STATE"], n_jobs=p["N_JOBS"], ratio="majority")),
    ("smote", SMOTE(random_state=p["RANDOM_STATE"], n_jobs=p["N_JOBS"])),
    ("smoteenn", SMOTEENN(random_state=p["RANDOM_STATE"], smote=SMOTE(random_state=p["RANDOM_STATE"], n_jobs=p["N_JOBS"]), enn=EditedNearestNeighbours(random_state=p["RANDOM_STATE"], n_jobs=p["N_JOBS"], ratio="majority"))),
    ("smotetl", SMOTETomek(random_state=p["RANDOM_STATE"], n_jobs=p["N_JOBS"], smote=SMOTE(random_state=p["RANDOM_STATE"], n_jobs=p["N_JOBS"]), tomek=TomekLinks(random_state=p["RANDOM_STATE"], n_jobs=p["N_JOBS"], ratio="majority")))
])

cattap = joblib.load("stratified_traintest.oh.pkl.xz")

for k, v in sampler.items():
    print("Start: " + k)
    newX, newY = v.fit_sample(cattap["X"].drop(['Accident_Index', 'Date_Time', 'Year'], axis=1), cattap["Y"])
    joblib.dump({"X": newX, "Y": newY}, "stratified_XY_train.oh." + k + ".pkl.xz")
    print("Count: " + str(Counter(newY)))
