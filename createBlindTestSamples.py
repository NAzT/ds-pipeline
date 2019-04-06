from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
from sklearn import datasets


class CreateBlindTestSamples:
    def create():
        for file in os.listdir('lib/blindedData'):
            os.remove(os.path.join('lib/blindedData', file))
        raw_data = datasets.load_boston()
        input_data = pd.DataFrame(raw_data.data)
        input_data.columns = raw_data.feature_names
        input_data['PRICE'] = raw_data.target

        X = input_data.drop('PRICE', axis=1)
        Y = input_data['PRICE']
        X_train, X_blind_test, y_train, y_blind_test = train_test_split(X, Y, test_size=0.3, random_state=0)
        column_head = pd.Index(['y']).append(X_train.columns)

        train = pd.DataFrame(np.column_stack([y_train, X_train]), columns=column_head)
        blind = pd.DataFrame(
            np.column_stack([y_blind_test, X_blind_test]), columns=column_head)

        train.to_csv('lib/blindedData/blindedTrainSample.csv', index=False)
        blind.to_csv('lib/blindedData/blindedTestSample.csv', index=False)
