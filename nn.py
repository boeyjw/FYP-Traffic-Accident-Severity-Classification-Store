"""
Something wrong with the dataset,
output is has variable minute loss changes but has constant accuracy
where all it predicts is class 3 (majority class)
"""
import pandas as pd
import numpy as np
import tap

from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.utils.class_weight import compute_class_weight
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras import regularizers
from keras import initializers
from keras.callbacks import TerminateOnNaN, EarlyStopping, CSVLogger

print('Init')
t = tap.modelling(ver_dir = '2.3')
data = t.load_tap(binarizer_mode = 'keras')
target = np.argmax(data.y, axis = 1)
class_weight = compute_class_weight('balanced', np.unique(target), target)
params = t.model_general_parameters(data_len = len(data.x), optimizer = 'sgd', loss = 'categorical_crossentropy', metrics = ['categorical_accuracy'], callbacks = [
                TerminateOnNaN(),
                EarlyStopping(patience=10)
                # CSVLogger('nn-hist-v1.csv')
            ], epochs=30, class_weight = {
                1: class_weight[0],
                2: class_weight[1],
                3: class_weight[2]
            })

X_train, X_test, Y_train, Y_test = train_test_split(data.x, data.y, test_size = params['test_size'], random_state = params['random_state'])

print('Fit')
model = Sequential()
model.add(Dense(len(t.x_cols) * 2, input_dim = len(t.x_cols), activation='relu', bias_initializer=initializers.glorot_uniform(), activity_regularizer=regularizers.l2()))
model.add(Dense(len(t.x_cols), activation='relu', activity_regularizer=regularizers.l2(), bias_initializer=initializers.glorot_uniform()))
model.add(Dense(4, activation='softmax'))
model.compile(optimizer = params['optimizer'], loss = params['loss'], metrics = params['metrics'])


hist = model.fit(X_train, Y_train, batch_size = params['batch_size'], epochs = params['epochs'], validation_split = params['validation_split'], callbacks = params['callbacks'], class_weight=params['class_weight'])

print('Predict')
y_pred = model.predict(X_test, batch_size=params['batch_size'])

metrics = tap.modelmetrics(binarizer=t.binarizer)
score = metrics.evaluate_model(np.argmax(Y_test, axis = 1), np.argmax(y_pred, axis=1), do_print=True)
