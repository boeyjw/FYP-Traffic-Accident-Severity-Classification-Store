import numpy as np
import pandas as pd

from collections import namedtuple

class dataload:
    def __init__(self, ver_dir, model_on, file_ver):
        self.__model_on = str(model_on).lower()
        if self.__model_on != 'train' or self.__model_on != 'test':
            raise TypeError('model_on argument only accepts train or test')
        dir = 'Imputation/' + ver_dir + '/' + self.__model_on + '/tap2005_2015-v2018.' + file_ver + '.' + self.__model_on + '.csv'
        
        self.tap = pd.read_csv(dir)
        self.data = self.load()
        self.rec_len = len(self.data.y)

    def load(self, drop_cols=[], target='Accident_Severity'):
        data = namedtuple('data', ['x', 'y'])
        data.x = self.tap.drop(target + drop_cols, axis=1)
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
        