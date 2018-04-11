import modelmetrics

from multiprocessing import cpu_count
from sklearn.externals import joblib

def get_constants(**kwargs):
    c = {
        "RANDOM_STATE": 123456789,
        "VALIDATION_SIZE": 0.2,
        "TEST_sIZE": 0.2,
        "TRAIN_SIZE": 0.8,
        "N_JOBS": cpu_count() -1
    }
    for k, v in kwargs.items():
        c[k] = v
    
    return c

def get_data_dir(impute_ver="2.5"):
    dirs = {
        "orig": "pointer/",
        "impute": "Imputation/" + impute_ver + "/",
    }

    return dirs

def fit_predict(estimator, X_train, Y_train, X_test, rfecv_cols, Y_test=None):
    y_pred = estimator.fit(X_train[rfecv_cols], Y_train).predict(X_test[rfecv_cols])

    if Y_test is not None:
        met = modelmetrics.metrics()
        return met.evaluate_model(y_true=Y_test, y_pred=y_pred, do_print=False)
    else:
        return y_pred
