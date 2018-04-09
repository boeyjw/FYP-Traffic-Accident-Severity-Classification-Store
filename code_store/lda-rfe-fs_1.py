import pandas as pd

from collections import OrderedDict
from sklearn.feature_selection import RFE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.externals import joblib

ver_dir = '2.4'

tap = pd.read_csv('acc2005_2016-v2018.' + ver_dir + '.csv').merge(
    pd.read_csv('veh2005_2016-v2018.' + ver_dir + '.csv'), on='Accident_Index'
).merge(
    pd.read_csv('cas2005_2016-v2018.' + ver_dir + '.csv'), on='Accident_Index'
).drop(['Accident_Index', 'Date_Time'], axis=1)

lda = OrderedDict([
    ('lda-rfe-svd', LinearDiscriminantAnalysis(solver='svd')),
    ('lda-rfe-lsqr', LinearDiscriminantAnalysis(solver='lsqr')),
    ('lda-rfe-eigen', LinearDiscriminantAnalysis(solver='eigen')),
    ('lda-rfe-lsqr-sh', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')),
    ('lda-rfe-eigen-sh', LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto'))
])
rfe_store = {}

for label, clf in lda.items():
    rfe = RFE(clf, n_features_to_select=20, step=1, verbose=1)
    rfe.fit(tap.drop('Accident_Severity', axis=1), tap['Accident_Severity'])
    rfe_store.update({
        label: rfe
    })

joblib.dump(rfe_store, 'tmp/lda-rfe-fs1.pkl')
