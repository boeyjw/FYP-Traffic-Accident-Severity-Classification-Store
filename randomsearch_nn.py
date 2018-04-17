import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from scipy.stats import randint as sp_randint
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
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

def create_nn(units=90, units_2=50, drop_rate=0.2, kernel_init="glorot_uniform", optimizer="adam"):
    nn = Sequential()
    # Manually change input_dim to match input shape
    nn.add(Dense(units=units, input_dim=70, activation="relu", kernel_initializer=kernel_init))
    nn.add(Dropout(rate=drop_rate))
    nn.add(Dense(units=units_2, activation="relu", kernel_initializer=kernel_init))
    nn.add(Dropout(rate=drop_rate))
    nn.add(Dense(units=3, activation="softmax"))
    nn.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["categorical_accuracy"])
    nn.summary()
    return nn
# def create_nn(units=90, units_2=50, kernel_reg=0.0001, kernel_init="glorot_uniform", optimizer="adam"):
#     nn = Sequential()
#     # Manually change input_dim to match input shape
#     nn.add(Dense(units=units, input_dim=70, activation="relu", kernel_regularizer=l2(kernel_reg), kernel_initializer=kernel_init))
#     nn.add(Dense(units=units_2, activation="relu", kernel_regularizer=l2(kernel_reg), kernel_initializer=kernel_init))
#     nn.add(Dense(units=3, activation="softmax"))
#     nn.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["categorical_accuracy"])
    # return nn

if __name__ == "__main__":
    print("Init")
    ext = ".cas.v2"
    params = modelparams.get_constants()
    sample = pd.read_csv("train/stratified_XY_train.oh.tlsmote" + ext + ".csv")

    X = sample.drop("Accident_Severity", axis=1).copy()
    scaler = StandardScaler()
    ordinal_cols = np.array(["Number_of_Casualties", "Number_of_Vehicles", "Speed_Limit", "Age_Band_of_Casualty"])
    ordinal_cols_mask = np.isin(ordinal_cols, X.columns)
    if any(ordinal_cols_mask):
        X[ordinal_cols[ordinal_cols_mask]] = scaler.fit_transform(X[ordinal_cols[ordinal_cols_mask]])
    Y = sample["Accident_Severity"].copy() - 1
    Xt, xv, Yt, yv = train_test_split(X, Y, stratify=Y, test_size=0.2)
    del sample, X, Y

    nn = KerasClassifier(build_fn=create_nn, epochs=75, batch_size=258, verbose=0)
    n_iter_torun = 1
    param_dist = {
        "units": sp_randint(int(np.round(Xt.shape[1] * 0.5)),  Xt.shape[1] + 50),
        "units_2": sp_randint(int(np.round(Xt.shape[1] * 0.5)),  Xt.shape[1]),
        # "kernel_reg": [0.01, 0.001, 0.0001],
        "kernel_init": ["glorot_uniform", "lecun_uniform", "glorot_normal", "lecun_normal"],
        "optimizer": ["adam", "nadam"],
        "batch_size": [128, 256, 512],
        "epochs": [75, 100, 150],
        "drop_rate": list(range(0.2, 0.5, 0.1))
    }
    fit_params = {
        "callbacks": [EarlyStopping(monitor="val_loss", patience=10)],
        "validation_data": (xv, to_categorical(yv))
    }
    rscv = RandomizedSearchCV(estimator=nn, n_iter=n_iter_torun, param_distributions=param_dist, scoring=["f1_macro", "recall_macro", "precision_macro", "accuracy"], refit="f1_macro", cv=StratifiedKFold(5), verbose=2, return_train_score=False, n_jobs=2, pre_dispatch=2, fit_params=fit_params)

    print("Search NN: " + ext)
    rscv.fit(Xt, Yt)
    cv_res = pd.DataFrame(rscv.cv_results_)
    print("Dump")
    cv_res.to_csv("search/nn/randomsearch_nn" + ext + ".csv", index=False)
    report(rscv.cv_results_, n_top=n_iter_torun)

