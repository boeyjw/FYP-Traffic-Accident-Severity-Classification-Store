import pandas as pd
import numpy as np

from collections import namedtuple
# Data & model manager
from sklearn.externals import joblib
from sklearn.preprocessing import LabelBinarizer
from keras.utils import to_categorical

# Metrics
from sklearn.metrics import accuracy_score, confusion_matrix
from imblearn.metrics import classification_report_imbalanced, sensitivity_specificity_support
from scipy.stats import ttest_rel

class tap:
    """
    Parent class for any classifier using the TAP dataset
    """
    def __init__(self, x_cols=None, rec_len=None):
        self.binarizer = {
            'mode': None,
            'obj': None
        }
        self.x_cols = x_cols
        self.rec_len = rec_len

    """
    Iteratively does a sklearn joblib dump for any sklearn output
    """
    def joblib_dumper(self, **kwargs):
        for f, o in kwargs.items():
            joblib.dump(o, str(f) + '.pkl')

class modelling(tap):
    """
    Traffic Accident Prediction data and model hyperparameter manager
    """
    def __init__(self, ver_dir, ver_file=None, use_current_dir=False, max_year=2015):
        # Get the directory right
        if ver_file is None:
            ver_file = ver_dir
        directory = './' if use_current_dir == True else 'Imputation/' + ver_dir + '/'
        # Import CSV data
        self.__data = pd.read_csv(directory + 'acc2005_' + str(max_year) + '-v2018.' + ver_file + '.imp.csv').merge(
            pd.read_csv(directory + 'veh2005_' + str(max_year) + '-v2018.' + ver_file + '.imp.csv'), on = 'Accident_Index', how = 'inner').drop(['Accident_Index', 'Date_Time'], axis = 1)
        # Initialise general pointers on the dataset
        super().__init__(x_cols=self.__data.columns.drop('Accident_Severity'), rec_len=len(self.__data))

    """
    Loads TAP dataset with selected target vector option
    """
    def load_tap(self, binarizer_mode=None):
        data = namedtuple('data', ['x', 'y'])
        target = self.__data['Accident_Severity']
        self.binarizer['name'] = binarizer_mode

        if binarizer_mode == 'sklearn':
            self.binarizer['obj'] = LabelBinarizer().fit(target)
            target = self.binarizer['obj'].transform(target)
        elif binarizer_mode == 'keras':
            target = to_categorical(target)

        return data(self.__data.drop('Accident_Severity', axis=1), target)

    """Dict to store all model hyperparameters
    """
    def model_general_parameters(self, random_state=500, batch_size=0.1, n_jobs=6, learning_rate=0.01, epochs=150, test_size=0.3, validation_split=0.1, **kwargs):
        params = {
            'random_state': random_state,
            'batch_size': int(np.ceil(self.rec_len * (1 - test_size) * batch_size)) if isinstance(batch_size, float) else batch_size,
            'n_jobs': n_jobs,
            'learning_rate': learning_rate,
            'epochs': epochs,
            'test_size': test_size,
            'validation_split': validation_split
        }
        for k, v in kwargs.items():
            params[k] = v

        return params

class modelmetrics(tap):
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
    def evaluate_model(self, y_true=None, y_pred=None, mode='weighted', name=None, do_print=False):
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
                print(k + ': ' + v, sep='')

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

