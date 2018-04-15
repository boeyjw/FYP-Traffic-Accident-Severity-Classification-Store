import numpy as np

from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from scipy.stats import randint as sp_randint
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from tap import modelparams

# Reference: http://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html#sphx-glr-auto-examples-model-selection-plot-randomized-search-py
# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                results['mean_test_score'][candidate],
                results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

def create_nn(input_dim):
    nn = Sequential()
    nn.add(Dense(units=int((input_dim + 3)/3), input_dim=input_dim, activation="relu"))
    nn.add(Dense(units=3, activation="softmax"))
    nn.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"])
    nn.summary()
    return nn

print("Init")
nn = KerasClassifier(build_fn=create_nn)
params = modelparams.get_constants()
n_iter_torun = 100
ext = ".cas"
param_dist = {

}
rscv = RandomizedSearchCV(estimator=nn, n_iter=n_iter_torun, param_distributions=param_dist, scoring="f1_micro", cv=StratifiedKFold(5), verbose=1, n_jobs=2)
scaler = StandardScaler()
ordinal_cols = np.array(["Number_of_Casualties", "Number_of_Vehicles", "Speed_Limit", "Age_Band_of_Casualty"])
sample = joblib.load("train/stratified_XY_train.oh.tlsmote.cas.pkl.xz")
ordinal_cols_mask = np.isin(ordinal_cols, sample["X"].columns)

X = sample["X"]
if any(ordinal_cols_mask):
    X[ordinal_cols[ordinal_cols_mask]] = scaler.fit_transform(X[ordinal_cols[ordinal_cols_mask]])
Y = to_categorical(sample["Y"] - 1)
del sample

print("Search RF")
rscv.fit(sample["X"], sample["Y"])
joblib.dump(rscv, "random_search_nn" + ext + ".pkl.xz")
report(rscv.cv_results_)

