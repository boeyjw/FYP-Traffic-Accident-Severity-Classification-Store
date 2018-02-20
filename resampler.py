from multiprocessing import Pool, Process, Lock, Manager

from tap import modelling, modelmetrics

from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

def rf_check_perf(x, y, name=None):
    rf = RandomForestClassifier(n_estimators=30, n_jobs=params['n_jobs'], random_state=params['random_state'])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=params['test_size'], random_state=params['random_state'])
    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_test)
    return metrics.evaluate_model(y_pred=y_pred, y_true=y_test, name=name, do_print=True)

def joblib_dump(x, y, postfix):
    joblib.dump(x, 'tmp/' + 'X_' + postfix + '_tst.pkl')
    joblib.dump(y, 'tmp/' + 'Y_' + postfix + '_tst.pkl')

def resample(sampler, name, datax, datay, res, lock):
    print(name)
    x_res, y_res = sampler.fit_sample(datax, datay)
    lock.acquire()
    res[name] = rf_check_perf(x_res, y_res, name=name)
    lock.release()
    joblib_dump(x_res, y_res, name)

model = modelling('2.5')
data = model.load_tap()
params = model.model_general_parameters(n_jobs=6, data_len=len(data.y))

metrics = modelmetrics()
res = Manager().dict([(clf, {}) for clf in ['default', 'tomek', 'smote', 'smotetomek']])

samplers = {
    'tomek': TomekLinks(n_jobs=params['n_jobs'], random_state=params['random_state']),
    'smote': SMOTE(n_jobs=params['n_jobs'], random_state=params['random_state'], kind='borderline1')
}

samplers['smotetomek'] = SMOTETomek(random_state=params['random_state'], smote=samplers['smote'], tomek=samplers['tomek'])

print('No sampling')
res['default'] = rf_check_perf(data.x[1:100000], data.y[1:100000])

lock = Lock()
pool = Pool(len(samplers))
for name, sampler in samplers:
    pool.apply_async(resample, args=(sampler, name, data.x[1:100000], data.y[1:100000], res, lock))

joblib.dump(res, 'tmp/' + 'sampling_results_tst.pkl')

