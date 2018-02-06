import pandas as pd
import numpy as np

from collections import namedtuple
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from keras.utils import to_categorical
class tap:
    __data = pd.DataFrame()
    x_cols = None
    rec_len = None
    """
    Traffic Accident Prediction data and hyperparameter manager
    """
    def __init__(self, ver_dir, ver_file = None, use_current_dir = False):
        # Get the directory right
        if ver_file is None:
            ver_file = ver_dir
        directory = './' if use_current_dir == True else 'Imputation/' + ver_dir + '/'
        # Import CSV data
        self.__data = pd.read_csv(directory + 'acc2005_2016-v2018.' + ver_file + '.imp.csv').merge(pd.read_csv(directory + 'veh2005_2016-v2018.' + ver_file + '.imp.csv'), on = 'Accident_Index', how = 'inner').drop(['Accident_Index', 'Date_Time'], axis = 1)
        # Initialise general pointers on the dataset
        self.x_cols = self.__data.columns.drop('Accident_Severity')
        self.rec_len = len(self.__data)

    def load_tap(self, binarizer_mode = 'sklearn'):
        data = namedtuple('data', ['x', 'y'])
        return data(self.__data.drop('Accident_Severity', axis=1),
        LabelBinarizer().fit_transform(self.__data['Accident_Severity']) if binarizer_mode == 'sklearn' else to_categorical(self.__data['Accident_Severity']))

    def model_general_parameters(self, data_len, random_state = 500, batch_size = 0.1, n_jobs = 6, learning_rate = 0.01, epochs = 150, test_size = 0.3, validation_split = 0.1, **kwargs):
        params = {
            'random_state': random_state,
            'batch_size': int(np.ceil(data_len * (1 - test_size) * batch_size)) if isinstance(batch_size, float) else batch_size,
            'n_jobs': n_jobs,
            'learning_rate': learning_rate,
            'epochs': epochs,
            'test_size': test_size,
            'validation_split': validation_split
        }
        for k, v in kwargs.items():
            params[k] = v

        return params

    def evaluate_model(self, y_true, y_pred, mode = 'weighted'):
        score = {
            'acc': accuracy_score(y_true, y_pred),
            'cm': confusion_matrix(y_true, y_pred),
            'report': classification_report(y_true, y_pred)
        }
        return score

