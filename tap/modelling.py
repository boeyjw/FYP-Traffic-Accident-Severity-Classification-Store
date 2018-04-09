import numpy as np
import pandas as pd

from collections import namedtuple

class dataload:
    def __init__(self, ver_dir, model_on=None, file_ver=None):
        self.__model_on = str(model_on).lower() if model_on is not None else None
        dir = 'Imputation/' + str(ver_dir) + '/'
        if self.__model_on is None:
            self.acc = pd.read_csv(dir + 'acc2005_2015-v2018.' + str(ver_dir) + '.imp.csv')
            self.veh = pd.read_csv(dir + 'veh2005_2015-v2018.' + str(ver_dir) + '.imp.csv')
            self.cas = pd.read_csv(dir + 'cas2005_2015-v2018.' + str(ver_dir) + '.imp.csv')
        else:
            dir += self.__model_on + '/tap2005_2015-v2018.' + str(file_ver) + '.' + self.__model_on + '.csv'

        self.tap = pd.read_csv(dir) if model_on is not None else self.acc.merge(self.veh, on='Accident_Index').merge(self.cas, on='Accident_Index')
        self.data = self.load() if model_on is None else self.load(target=None)
        self.rec_len = len(self.data.y) if model_on is None else len(self.tap)

    def load(self, drop_cols=[], target='Accident_Severity'):
        data = namedtuple('data', ['x', 'y'])
        if target is not None:
            data.x = self.tap.drop([target] + drop_cols, axis=1)
            data.y = self.tap.loc[target]

        return data

    def get(self):
        return self.data

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
