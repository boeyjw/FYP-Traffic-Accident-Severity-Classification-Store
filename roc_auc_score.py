import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from multiprocessing import cpu_count
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.externals import joblib
from keras.models import load_model
from keras.utils import to_categorical
from scipy import interp

# Reference: http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html

RANDOM_STATE = 123456789
N_JOBS=cpu_count() - 1 # Leave 1 thread for system use (extremely important during thrashing)
is_rf = False # SET THIS TO FALSE IF NN
ext = ".nocas.v2"
fn = "roc_auc_score-all.oh.tlsmote" + ext + ".pkl.xz"
plt_title = 'Artificial Neural Network ROC Curve for No Casualty Features'

print("Init")
# Import training dataset
# X, y = joblib.load("stratified_X_train.pkl.z"), joblib.load("stratified_Y_train.pkl.z")
sample = joblib.load("train/stratified_XY_train.oh.tlsmote" + ext + ".pkl.xz")
cattap = joblib.load("test/stratified_traintest.oh.pkl.xz")
x, y = cattap["x"][sample["X2"].columns].copy(), cattap["y"].copy()
del sample, cattap

# NN preprocessing
if not is_rf:
    scaler = StandardScaler()
    ordinal_cols = np.array(["Number_of_Casualties", "Number_of_Vehicles", "Speed_Limit", "Age_Band_of_Casualty"])
    ordinal_cols_mask = np.isin(ordinal_cols, x.columns)
    if any(ordinal_cols_mask):
        x[ordinal_cols[ordinal_cols_mask]] = scaler.fit_transform(x[ordinal_cols[ordinal_cols_mask]])

# Binarize the test label
lb = LabelBinarizer()
y_test = lb.fit_transform(y)
n_classes = y_test.shape[1]

print("Predict Proba")
# Learn to predict each class against the other
classifier = joblib.load("final/rf.final" + ext + ".pkl.xz")["model"] if is_rf else load_model("final/nn.model.val.final" + ext + ".h5")
y_score = classifier.predict_proba(x)

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
joblib.dump({"fpr":fpr, "tpr":tpr, "roc_auc":roc_auc}, "final/" + fn)

print("Plot: " + ext)
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
plt.title(plt_title)
plt.legend(loc="lower right")
plt.show()
