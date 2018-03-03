from tap import modelling

from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

rfecv = RFECV(RandomForestClassifier(n_estimators=30, n_jobs=6, verbose=1), verbose=1)
pre_model = modelling('2.5')
data = pre_model.load_tap()
drop_cols = ['Day_of_Week', 'Is_Holiday', 'Year', 'Month', 'Hour']

selector = rfecv.fit(data.x.drop(drop_cols, axis=1), data.y)
joblib.dump(selector, 'tmp/rfecv-fs2-v2018-2.5.pkl')
