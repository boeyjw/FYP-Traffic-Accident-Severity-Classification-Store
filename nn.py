"""
Something wrong with the dataset,
output is has variable minute loss changes but has constant accuracy
where all it predicts is class 3 (majority class)
"""
import pandas as pd
import numpy as np

from tap import tap
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.callbacks import TerminateOnNaN, EarlyStopping, TensorBoard, CSVLogger

print('Init')
tap = tap(ver_dir = '2.3')
data = tap.load_tap(binarizer_mode = 'keras')
params = tap.model_general_parameters(data_len = len(data.x), optimizer = 'sgd', loss = 'categorical_crossentropy', metrics = ['categorical_accuracy'], callbacks = [
                TerminateOnNaN(),
                EarlyStopping(patience=10),
                CSVLogger('nn-hist-v1.csv')
            ])

X_train, X_test, Y_train, Y_test = train_test_split(data.x, data.y, test_size = params['test_size'], random_state = params['random_state'])

print('Fit')
model = Sequential()
model.add(Dense(len(tap.x_cols) * 2, input_dim = len(tap.x_cols), activation='elu'))
model.add(Dense(len(tap.x_cols), activation='elu'))
model.add(Dense(4, activation='softmax'))
model.compile(optimizer = params['optimizer'], loss = params['loss'], metrics = params['metrics'])

hist = model.fit(X_train, Y_train, batch_size = params['batch_size'], epochs = params['epochs'], validation_split = params['validation_split'], callbacks = params['callbacks'])
model.save('nn_model-v1.h5')

print('Predict')
score = model.evaluate(X_test, Y_test)

print('Test score: ', score[0])
print('Test Accuracy: ', score[1])
