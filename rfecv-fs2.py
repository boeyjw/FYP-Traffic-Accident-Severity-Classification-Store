import tap

from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

rfecv = RFECV(RandomForestClassifier(n_estimators=30, n_jobs=3))
pre_model = tap.modelling('2.3')
data = pre_model.load_tap()

selector = rfecv.fit(data.x, data.y)
joblib.dump(selector, 'rfecv-fs2-v1.pkl')
