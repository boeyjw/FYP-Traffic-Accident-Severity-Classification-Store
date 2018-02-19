from tap import modelling, modelmetrics
from collections import OrderedDict

from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

model = modelling('2.5')
data = model.load_tap()
data.x = data.x.loc[1:10000]
data.y = data.y.loc[1:10000]
params = model.model_general_parameters(n_jobs=6, data_len=len(data.y))

metrics = modelmetrics()
res = OrderedDict([(clf, {}) for clf in ['default', 'tomek', 'smote', 'smotetomek']])

def rf_check_perf(x, y):
    rf = RandomForestClassifier(n_estimators=30, n_jobs=params['n_jobs'], random_state=params['random_state'])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=params['test_size'], random_state=params['random_state'])
    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_test)
    return metrics.evaluate_model(y_pred=y_pred, y_true=y_test)

def joblib_dump(x, y, postfix):
    joblib.dump(x, 'tmp/' + 'X_' + postfix + '.pkl')
    joblib.dump(y, 'tmp/' + 'Y_' + postfix + '.pkl')

print('No sampling')
res['default'] = rf_check_perf(data.x, data.y)

print('Tomek')
tomek = TomekLinks(n_jobs=params['n_jobs'], random_state=params['random_state'])
x_res, y_res = tomek.fit_sample(data.x, data.y)
res['tomek'] = rf_check_perf(x_res, y_res)
joblib_dump(x_res, y_res, 'tomek')
del x_res, y_res, tomek

print('SMOTE')
smote = SMOTE(n_jobs=params['n_jobs'], random_state=params['random_state'], kind='borderline1')
x_res, y_res = smote.fit_sample(data.x, data.y)
res['smote'] = rf_check_perf(x_res, y_res)
joblib_dump(x_res, y_res, 'smote')
del x_res, y_res, smote

print('SMOTETomek')
smote_tomek = SMOTETomek(random_state=params['random_state'], smote=smote, tomek=tomek, n_jobs=params['n_jobs'])
x_res, y_res = smote_tomek.fit_sample(data.x, data.y)
res['smotetomek'] = rf_check_perf(x_res, y_res)
joblib_dump(x_res, y_res, 'smotetomek')
del x_res, y_res, smote_tomek

joblib.dump(res, 'tmp/' + 'sampling_results.pkl')

