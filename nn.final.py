import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tap import modelmetrics, modelparams

def build_nn(units, units_2, input_dim):
    nn = Sequential()
    nn.add(Dense(units=units, input_dim=input_dim, activation="relu", kernel_initializer="lecun_normal"))
    nn.add(Dropout(rate=0.2))
    nn.add(Dense(units=units_2, activation="relu", kernel_initializer="lecun_normal"))
    nn.add(Dropout(rate=0.2))
    nn.add(Dense(units=3, activation="softmax"))
    nn.compile(optimizer="nadam", loss="categorical_crossentropy", metrics=["categorical_accuracy"])
    return nn

def nn_predict(nn, x, batch_size):
    return [np.argmax(v) + 1 for v in nn.predict(x, batch_size=batch_size)]

print("Init")
ext = ".cas.v2"
params = modelparams.get_constants()
met = modelmetrics.metrics()
scaler = StandardScaler()
test = joblib.load("test/stratified_traintest.oh.pkl.xz")
epochs = 200
batch_size = 512

ordinal_cols = np.array(["Number_of_Casualties", "Number_of_Vehicles", "Speed_Limit", "Age_Band_of_Casualty"])
x = test["x"].copy() # test x
y = test["y"].copy() # test y
del test
ordinal_cols_mask = np.isin(ordinal_cols, x.columns)
if any(ordinal_cols_mask):
    x[ordinal_cols[ordinal_cols_mask]] = scaler.fit_transform(x[ordinal_cols[ordinal_cols_mask]])

print(ext)
train = joblib.load("train/stratified_XY_train.oh.tlsmote" + ext + ".pkl.xz")
X = train["X2"].copy()
Y = train["Y"].copy()
del train
ordinal_cols_mask = np.isin(ordinal_cols, X.columns)
if any(ordinal_cols_mask):
    X[ordinal_cols[ordinal_cols_mask]] = scaler.fit_transform(X[ordinal_cols[ordinal_cols_mask]])
Xt, xv, Yt, yv = train_test_split(X, Y, test_size=0.1, stratify=Y)
del X, Y
nn = build_nn(int(np.round(Xt.shape[1] * 2.5)), Xt.shape[1] * 2, Xt.shape[1])
hist = nn.fit(Xt, to_categorical(Yt - 1), batch_size=batch_size, epochs=epochs, validation_data=(xv, to_categorical(yv - 1)))
y_pred = nn_predict(nn, x[Xt.columns], batch_size)
res = met.evaluate_model(y_true=y, y_pred=y_pred, mode="macro", name=ext, store_y_pred=True)
joblib.dump({
    "result": res,
    "history": hist.history
}, "final/nn.val.final" + ext + ".pkl.xz")
nn.save("final/nn.model.val.final" + ext + ".h5")


