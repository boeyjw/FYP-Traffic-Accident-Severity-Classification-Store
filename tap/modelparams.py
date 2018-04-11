from multiprocessing import cpu_count
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from tap import modelmetrics

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

def fit_predict(estimator, X_train, Y_train, X_test, rfecv_cols=None, Y_test=None):
    y_pred = estimator.fit(X_train if rfecv_cols is None else X_train[rfecv_cols], Y_train).predict(X_test if rfecv_cols is None else X_test[rfecv_cols])

    if Y_test is not None:
        met = modelmetrics.metrics()
        return met.evaluate_model(y_true=Y_test, y_pred=y_pred, do_print=False)
    else:
        return y_pred

def validation_split(X_train, Y_train, rfecv_cols=None, random_state=None, val_size=0.2):
    Xt, xt, Yt, yt = train_test_split(X_train if rfecv_cols is None else X_train[rfecv_cols], Y_train, test_size=val_size, random_state=random_state, stratify=Y_train)
    return Xt, xt, Yt, yt
