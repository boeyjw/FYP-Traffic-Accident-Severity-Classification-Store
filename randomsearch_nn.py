import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from keras.regularizers import l2
from scipy.stats import randint as sp_randint
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from tap import modelparams

# Reference: http://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html#sphx-glr-auto-examples-model-selection-plot-randomized-search-py
# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_f1_micro'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                results['mean_test_f1_micro'][candidate],
                results['std_test_f1_micro'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

def create_nn(units=125, kernel_reg=0.0001, kernel_init="glorot_uniform", optimizer="adam"):
    nn = Sequential()
    # Manually change input_dim to match input shape
    nn.add(Dense(units=units, input_dim=125, activation="relu", kernel_regularizer=l2(kernel_reg), kernel_initializer=kernel_init))
    nn.add(Dense(units=3, activation="softmax"))
    nn.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["categorical_accuracy"])

    return nn
if __name__ == "__main__":
    print("Init")
    ext = ".cas"
    params = modelparams.get_constants()
    sample = pd.read_csv("train/stratified_XY_train.oh.tlsmote" + ext + ".csv")

    X = sample.drop("Accident_Severity", axis=1)
    scaler = StandardScaler()
    ordinal_cols = np.array(["Number_of_Casualties", "Number_of_Vehicles", "Speed_Limit", "Age_Band_of_Casualty"])
    ordinal_cols_mask = np.isin(ordinal_cols, X.columns)
    if any(ordinal_cols_mask):
        X[ordinal_cols[ordinal_cols_mask]] = scaler.fit_transform(X[ordinal_cols[ordinal_cols_mask]])
    Y = sample["Accident_Severity"] - 1
    del sample

    nn = KerasClassifier(build_fn=create_nn, epochs=50, batch_size=128, verbose=0)
    n_iter_torun = 20
    param_dist = {
        "units": sp_randint(int(np.round(X.shape[1] * 0.5)),  X.shape[1] + 50),
        "kernel_reg": [0.01, 0.001, 0.0001],
        "kernel_init": ["glorot_uniform", "lecun_uniform", "glorot_normal", "lecun_normal"],
        "optimizer": ["adam", "nadam"],
        "epochs": [75, 100, 150],
        "batch_size": [128, 256, 512]
    }
    rscv = RandomizedSearchCV(estimator=nn, n_iter=n_iter_torun, param_distributions=param_dist, scoring=["f1_micro", "recall_micro", "precision_micro", "accuracy"], refit="f1_micro", cv=StratifiedKFold(5), verbose=2, return_train_score=False, n_jobs=2, pre_dispatch=2)

    print("Search RF: " + ext)
    rscv.fit(X, Y)
    cv_res = pd.DataFrame(rscv.cv_results_)
    print("Dump")
    cv_res.to_csv("search/nn/randomsearch_nn" + ext + ".csv", index=False)
    rscv.best_estimator_.model.save("search/nn/" + "best_nn" + ext + ".randomsearch.h5")
    report(rscv.cv_results_, n_top=n_iter_torun)

