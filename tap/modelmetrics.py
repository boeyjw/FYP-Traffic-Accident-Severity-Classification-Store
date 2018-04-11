import numpy as np

from sklearn.metrics import confusion_matrix, accuracy_score
from imblearn.metrics import sensitivity_score, classification_report_imbalanced, sensitivity_specificity_support
from scipy.stats import ttest_rel

class metrics():
    """Utility class that provides metrics for model evaluation and comparison
    """
    def __init__(self, binarizer=None, y_true=None, y_pred=None):
        self.binarizer = binarizer
        self.y_true = y_true
        self.y_pred = y_pred
        if self.binarizer is not None:
            if y_true is not None:
                self.y_true = self.__reverse_binarizer(y_true)
            if y_pred is not None:
                self.y_pred = self.__reverse_binarizer(y_pred)

    """sklearn does not support vector target (multiclass), reverses the transformation to integer
    """
    def __reverse_binarizer(self, arr):
        return np.argmax(arr, axis = 1) if self.binarizer == 'keras' else self.binarizer['obj'].reverse_transform(arr)

    """Evaluates model accuracy, precision, recall, f1 and confusion matrix
    """
    def evaluate_model(self, y_true=None, y_pred=None, mode='macro', name=None, do_print=False):
        if y_true is None:
            y_true = self.y_true
        if y_pred is None:
            y_pred = self.y_pred
        sss = sensitivity_specificity_support(y_true, y_pred, average=mode)
        score = {
            'name': name,
            'accuracy': accuracy_score(y_true, y_pred),
            'confusion_matrix': confusion_matrix(y_true, y_pred),
            'report': classification_report_imbalanced(y_true, y_pred),
            'sensitivity': sss[0],
            'specificity': sss[1],
            'support': sss[2]
        }

        if do_print == True:
            self.parse_evaluate_model_print(score)
        return score

    @staticmethod
    def parse_evaluate_model_print(eval_res):
        for k, v in eval_res.items():
            if v is not None:
                print(k + ':\n{}\n'.format(v), sep='')

    """Student T-test to compare the significance between models
    """
    def ttest2(self, y_pred_1, y_pred_2, do_print=False, name=None):
        ttest = {'ttest': ttest_rel(y_pred_1, y_pred_2)}
        if name is not None:
            ttest['name'] = name
            if do_print == True:
                print(ttest['name'])
        if do_print == True:
            print('Statistics: {}\np-value: {}'.format(ttest['ttest'][0], ttest['ttest'][1]))

        return ttest
