import pandas as pd

from collections import namedtuple
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from keras.utils import to_categorical
class tap:
    def load_tap(self, ver_dir, ver_file = None, binarizer_mode = 'sklearn', use_current_dir = False):
        if ver_file is None:
            ver_file = ver_dir
        directory = './' if use_current_dir == True else 'Imputation/' + ver_dir + '/'
        tap = pd.read_csv(directory + 'acc2005_2016-v2018.' + ver_file + '.imp.csv').merge(pd.read_csv(directory + 'veh2005_2016-v2018.' + ver_file + '.imp.csv'), on = 'Accident_Index', how = 'inner').drop(['Accident_Index', 'Date_Time'], axis = 1)

        Data = namedtuple('Data', ['x', 'y'])
        return Data(tap.drop('Accident_Severity', axis=1), LabelBinarizer().fit_transform(tap['Accident_Severity']) if binarizer_mode == 'sklearn' else to_categorical(tap['Accident_Severity']))

    def model_general_parameters(self, random_state = 500, batch_size = 512000, n_jobs = 6, learning_rate = 0.01, epochs = 150, test_size = 0.3, validation_split = 0.1, **kwargs):
        pass

    def evaluate_model(self, y_true, y_pred, mode = 'weighted'):
        pass

