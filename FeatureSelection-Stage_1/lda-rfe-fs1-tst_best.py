import pandas as pd
import numpy as np

from collections import OrderedDict
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support

RANDOM_STATE = 500

# Init
tap = pd.read_csv('acc2005_2016-v2018.2.1.csv').merge(pd.read_csv('veh2005_2016-v2018.2.1.csv'), on='Accident_Index').drop(['Accident_Index', 'Date_Time'], axis=1)
fs1 = joblib.load('lda-rfe-fs1.pkl')
rf_clf = RandomForestClassifier(n_estimators=30, n_jobs=6, random_state=RANDOM_STATE, verbose=1)

X, Y = tap.drop('Accident_Severity', axis=1), tap['Accident_Severity']

y_score = OrderedDict([(label, {}) for label, _ in fs1.items()])

for label, fs in fs1.items():
    print(label)
    X_train, X_test, Y_train, Y_test = train_test_split(X.loc[:,fs.support_], Y, test_size=0.3, random_state=RANDOM_STATE)
    rf_clf.fit(X_train, Y_train)
    y_pred = rf_clf.predict(X_test)
    y_score[label] = {
        'CM': confusion_matrix(Y_test, y_pred),
        'ACC': accuracy_score(Y_test, y_pred),
        'PRF1': precision_recall_fscore_support(Y_test, y_pred, average='macro')
    }

joblib.dump(y_score, 'lda-rfe-fs1-res.pkl')

