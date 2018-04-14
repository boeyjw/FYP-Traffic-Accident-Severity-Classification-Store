from collections import OrderedDict, Counter
from sklearn.externals import joblib
from tap import modelparams
from imblearn.under_sampling import TomekLinks, EditedNearestNeighbours
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN, SMOTETomek

p = modelparams.get_constants()
# cattap = joblib.load("test/stratified_traintest.oh.pkl.xz")

## General approach undersampling-oversampling
# sampler = OrderedDict([
#     ("tl", TomekLinks(random_state=p["RANDOM_STATE"], n_jobs=p["N_JOBS"], ratio="majority")),
#     ("enn", EditedNearestNeighbours(random_state=p["RANDOM_STATE"], n_jobs=p["N_JOBS"], ratio="majority")),
#     ("smote", SMOTE(random_state=p["RANDOM_STATE"], n_jobs=p["N_JOBS"])),
#     ("smoteenn", SMOTEENN(random_state=p["RANDOM_STATE"], smote=SMOTE(random_state=p["RANDOM_STATE"], n_jobs=p["N_JOBS"]), enn=EditedNearestNeighbours(random_state=p["RANDOM_STATE"], n_jobs=p["N_JOBS"], ratio="majority"))),
#     ("smotetl", SMOTETomek(random_state=p["RANDOM_STATE"], n_jobs=p["N_JOBS"], smote=SMOTE(random_state=p["RANDOM_STATE"], n_jobs=p["N_JOBS"]), tomek=TomekLinks(random_state=p["RANDOM_STATE"], n_jobs=p["N_JOBS"], ratio="majority")))
# ])


# for k, v in sampler.items():
#     print("Start: " + k)
#     newX, newY = v.fit_sample(cattap["X"].drop(['Accident_Index', 'Date_Time', 'Year'], axis=1), cattap["Y"])
#     joblib.dump({"X": newX, "Y": newY}, "stratified_XY_train.oh." + k + ".pkl.xz")
#     print("Count: " + str(Counter(newY)))

## Undersample first then oversample
# undersampler = OrderedDict([
#     ("ennsmote", EditedNearestNeighbours(random_state=p["RANDOM_STATE"], n_jobs=p["N_JOBS"], ratio="all")),
#     ("tlsmote", TomekLinks(random_state=p["RANDOM_STATE"], n_jobs=p["N_JOBS"], ratio="all"))
# ])
# smote = SMOTE(random_state=p["RANDOM_STATE"], n_jobs=p["N_JOBS"])

# for k, v in undersampler.items():
#     print("Start: " + k)
#     newX, newY = v.fit_sample(cattap["X"].drop(['Accident_Index', 'Date_Time', 'Year'], axis=1), cattap["Y"]) # Data cleaning
#     newX, newY = smote.fit_sample(newX, newY) # SMOTE
#     joblib.dump({"X": newX, "Y": newY}, "stratified_XY_train.oh." + k + ".pkl.xz")
#     print("Count: " + str(Counter(newY)))

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

