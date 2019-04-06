from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import pandas as pd
import os


class CreateModels:
    @staticmethod
    def create(model):
        for file in os.listdir('lib/models'):
            os.remove(os.path.join('lib/models', file))
        train = pd.read_csv('lib/blindedData/blindedTrainSample.csv')

        X_train = train.drop(['y'], axis=1)
        y_train = train['y']

        kf = KFold(n_splits=3)
        train_index, test_index = next(kf.split(X_train))

        X_train_validate, X_test_validate = X_train.iloc[train_index, :], X_train.iloc[test_index, :]
        y_train_validate, y_test_validate = y_train.iloc[train_index], y_train.iloc[test_index]

        model.fit(X_train_validate, y_train_validate)
        y_pred_fit = model.predict(X_train_validate)
        y_pred_validate = model.predict(X_test_validate)
        fit_score = mean_squared_error(y_train_validate, y_pred_fit)
        validate_score = mean_squared_error(y_test_validate, y_pred_validate)
        print("%s: fit mse = %.2f and validate mse = %.2f" % (model.__class__.__name__, fit_score, validate_score))
