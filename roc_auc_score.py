import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from multiprocessing import cpu_count
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelBinarizer
from sklearn.externals import joblib
from scipy import interp

# Reference: http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html

RANDOM_STATE = 123456789
N_JOBS=cpu_count() - 1 # Leave 1 thread for system use (extremely important during thrashing)
fn = "roc_auc_score-all.oh.tlsmote.cas.pkl.xz"

print("Init")
# Import training dataset
# X, y = joblib.load("stratified_X_train.pkl.z"), joblib.load("stratified_Y_train.pkl.z")
sample = joblib.load("train/stratified_XY_train.oh.tlsmote.cas.pkl.xz")
cattap = joblib.load("test/stratified_traintest.oh.pkl.xz")

print("Validation set")
# Split out validation set
# X_train, X_test, Y_train, Y_test = train_test_split(sample["X"], sample["Y"], test_size=0.2, random_state=RANDOM_STATE, stratify=sample["Y"])
X_train, X_test, Y_train, Y_test = sample["X"], cattap["x"][sample["X"].columns], sample["Y"],  cattap["y"]
print(X_train.columns)
print(X_test.columns)
# # Load selected features
# rfecv = joblib.load("rfecv_withCAS-res.pkl")
# X_train, X_test, Y_train, Y_test = train_test_split(X[rfecv["after_sel_cols"]], y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

# Binarize the test label
lb = LabelBinarizer()
y_test = lb.fit_transform(Y_test)
n_classes = y_test.shape[1]

print("Train")
# Learn to predict each class against the other
classifier = RandomForestClassifier(n_estimators=50, n_jobs=N_JOBS, random_state=RANDOM_STATE)
y_score = classifier.fit(X_train, Y_train).predict_proba(X_test)

print("Compute ROC")
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

print("dump")
joblib.dump({"fpr":fpr, "tpr":tpr, "roc_auc":roc_auc}, fn)

print("Plot")
# Plot all ROC curves
plt.figure()
lw = 2
plt.plot(fpr["micro"], tpr["micro"],
        label='micro-average ROC curve (area = {0:0.2f})'
            ''.format(roc_auc["micro"]),
        color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
        label='macro-average ROC curve (area = {0:0.2f})'
            ''.format(roc_auc["macro"]),
        color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
            label='ROC curve of class {0} (area = {1:0.2f})'
            ''.format(i + 1, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Forest ROC Curve with All Features')
plt.legend(loc="lower right")
plt.show()
