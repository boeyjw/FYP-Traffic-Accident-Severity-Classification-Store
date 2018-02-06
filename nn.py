"""
Something wrong with the dataset,
output is has variable minute loss changes but has constant accuracy
where all it predicts is class 3 (majority class)
"""
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils import to_categorical
from keras.callbacks import TerminateOnNaN, EarlyStopping, TensorBoard, CSVLogger

print('Init')
RANDOM_STATE = 123456789
BATCH_SIZE = 2048000
tap = pd.read_csv('Imputation/2.3/acc2005_2016-v2018.2.3.imp.csv').merge(pd.read_csv('Imputation/2.3/veh2005_2016-v2018.2.3.imp.csv'), on = 'Accident_Index', how = 'inner').drop(['Accident_Index', 'Date_Time'], axis = 1)
sample_size = sum(tap.loc[tap['Accident_Severity'] == 1, 'Accident_Severity'])

X_train, X_test, Y_train, Y_test = train_test_split(tap.drop('Accident_Severity', axis=1), to_categorical(tap['Accident_Severity']), test_size = 0.3, random_state = RANDOM_STATE)

input_dim = len(tap.columns.drop('Accident_Severity'))

print('Fit')
model = Sequential()
model.add(Dense(input_dim * 2, input_dim = input_dim, activation='sigmoid'))
model.add(Dense(input_dim, activation='sigmoid'))
model.add(Dense(4, activation='softmax'))
model.compile(optimizer = 'sgd', loss = 'categorical_crossentropy', metrics=['categorical_accuracy'])

callbacks = [
                TerminateOnNaN(),
                EarlyStopping(patience=10),
                CSVLogger('nn-hist-v1.csv')
            ]
hist = model.fit(X_train, Y_train, batch_size = BATCH_SIZE, epochs = 30, validation_split = 0.1, callbacks = callbacks)
model.save('nn_model-v1.h5')

print('Predict')
score = model.evaluate(X_test, Y_test)

print('Test score: ', score[0])
print('Test Accuracy: ', score[1])
