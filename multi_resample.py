import multiprocessing

from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

from tap import modelling, modelmetrics

def random_forest_test(x, y, params, name=None):
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=params['test_size'], random_state=params['random_state'])
    rf = RandomForestClassifier(n_estimators=30, n_jobs=params['n_jobs'], random_state=params['random_state'])
    rf.fit(X_train, Y_train)
    return modelmetrics(y_true=Y_test, y_pred=rf.predict(X_test)).evaluate_model(name=name, do_print=True)

def joblib_dump(x, y, name):
    joblib.dump(value=x, filename='tmp/X_' + name + '.pkl.ignore', compress=True)
    joblib.dump(value=y, filename='tmp/Y_' + name + '.pkl.ignore', compress=True)

def resample(x, y, params, res, lock, sampler, name):
    print('Starting: ' + multiprocessing.current_process().name)
    X_res, Y_res = sampler.fit_sample(x, y)

    lock.acquire()
    print('Testing: ' + multiprocessing.current_process().name)
    res[name] = random_forest_test(x=X_res, y=Y_res, params=params, name=name)
    lock.release()
    
    print('Dumping: ' + multiprocessing.current_process().name)
    joblib_dump(X_res, Y_res, name)
    
    print('Exiting: ' + multiprocessing.current_process().name)

if __name__ == "__main__":
    model = modelling('2.5')
    data = model.load_tap()
    params = model.model_general_parameters()

    res = multiprocessing.Manager().dict([
        (name, {}) for name in ['default', 'tomek', 'smote', 'smotetomek']
    ])
    samplers = {
        'tomek': TomekLinks(random_state=params['random_state'], n_jobs=params['n_jobs']),
        'smote': SMOTE(random_state=params['random_state'], kind='borderline1', n_jobs=params['n_jobs']),
    }
    samplers['smotetomek'] = SMOTETomek(random_state=params['random_state'], n_jobs=params['n_jobs'], tomek=samplers['tomek'], smote=samplers['smote'])

    lock = multiprocessing.Lock()
    procs = [
        multiprocessing.Process(target=resample, name=name, args=(data.x, data.y, params, res, lock, sampler, name))
        for name, sampler in samplers.items()
    ]

    res['default'] = random_forest_test(data.x, data.y, params, name='benchmark')
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    
    joblib.dump(res, 'tmp/resampling_res-v2018.2.5.pkl')
