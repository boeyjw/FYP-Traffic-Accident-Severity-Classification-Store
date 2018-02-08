import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import chi2
from tap import tap, modelmetrics

print('init')
tap = tap('2.3')

data = tap.load_tap()
class_weight = compute_class_weight('balanced', np.unique(data.y), data.y)
params = tap.model_general_parameters(data_len = len(data.x), n_estimators = 50, n_jobs = 3, class_weight = {
    1: class_weight[0],
    2: class_weight[1],
    3: class_weight[2]
})

print('fit & predict 20 features')
X_train, X_test, Y_train, Y_test = train_test_split(data.x, data.y, test_size=params['test_size'], random_state=params['random_state'])
rf = RandomForestClassifier(n_estimators=params['n_estimators'], n_jobs=params['n_jobs'], random_state=params['random_state'], class_weight=params['class_weight'], verbose=1)
rf.fit(X_train, Y_train)
y_pred = rf.predict(X_test)
del rf

rf = RandomForestClassifier(n_estimators=params['n_estimators'], n_jobs=params['n_jobs'], random_state=params['random_state'])
rf.fit(X_train, Y_train)
y_pred_alt = rf.predict(X_test)
del rf

print('fit & predict chi features')
chi = chi2(data.x, data.y)
chi_cols = [col for col, s, p in zip(tap.x_cols, chi[0], chi[1]) if p > 0]
X_train, X_test, _, _ = train_test_split(data.x.loc[:, chi_cols], data.y, test_size=params['test_size'], random_state=params['random_state'])
rf = RandomForestClassifier(n_estimators=params['n_estimators'], n_jobs=params['n_jobs'], random_state=params['random_state'], class_weight=params['class_weight'], verbose=1)
rf.fit(X_train, Y_train)
y_pred_chi = rf.predict(X_test)
del rf

rf = RandomForestClassifier(n_estimators=params['n_estimators'], n_jobs=params['n_jobs'], random_state=params['random_state'])
rf.fit(X_train, Y_train)
y_pred_alt_chi = rf.predict(X_test)
del rf

metrics = modelmetrics()
print('evaluate')
y_score = metrics.evaluate_model(Y_test, y_pred, name = 'Random Forest with Class Weight on 20 features', do_print=True)
y_score_alt = metrics.evaluate_model(Y_test, y_pred_alt, name = 'Random Forest without Class Weight on 20 features', do_print=True)
metrics.ttest2(y_pred, y_pred_alt, do_print=True, name='RF 20 features')

y_score_chi = metrics.evaluate_model(Y_test, y_pred, name = 'Random Forest with Class Weight on CHI features', do_print=True)
y_score_alt_chi = metrics.evaluate_model(Y_test, y_pred_alt, name = 'Random Forest without Class Weight CHI features', do_print=True)
metrics.ttest2(y_pred_chi, y_pred_alt_chi, do_print=True, name='RF CHI features')

metrics.ttest2(y_pred, y_pred_chi, do_print=True, name='Compare RF 20 no class weight vs CHI no class weight')
metrics.ttest2(y_pred, y_pred_alt_chi, do_print=True, name='Compare RF 20 no class weight vs CHI with class weight')
metrics.ttest2(y_pred_alt, y_pred_chi, do_print=True, name='Compare RF 20 with class weight vs CHI no class weight')
metrics.ttest2(y_pred_alt, y_pred_alt_chi, do_print=True, name='Compare RF 20 with class weight vs CHI with class weight')
