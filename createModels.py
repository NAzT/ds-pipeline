from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, RandomizedSearchCV
import pickle
import pandas as pd
import os


class CreateModels:
    @staticmethod
    def create(model, param):
        cur_model_file = os.path.join('lib/models', model.__class__.__name__ + '.pickle')
        if os.path.exists(cur_model_file):
            os.remove(cur_model_file)
        train = pd.read_csv('lib/blindedData/blindedTrainSample.csv')
        X_train = train.drop(['y'], axis=1)
        y_train = train['y']

        blind_test = pd.read_csv('lib/blindedData/blindedTestSample.csv')
        X_blind_test = blind_test.drop(['y'], axis=1)
        y_blind_test = blind_test['y']
        kf = KFold(n_splits=10)
        pipeline = Pipeline(steps=[('norm', StandardScaler()), ('model', model)])

        optimizedModel = RandomizedSearchCV(pipeline, param, cv=kf, scoring='neg_mean_squared_error')
        optimizedModel.fit(X_train, y_train)

        print(optimizedModel.best_params_)

        y_pred_fit = optimizedModel.predict(X_train)
        y_pred_test = optimizedModel.predict(X_blind_test)

        fit_score = mean_squared_error(y_train, y_pred_fit)
        test_score = mean_squared_error(y_pred_test, y_blind_test)

        print("%s: fit mse = %.2f and test mse = %.2f" % (model.__class__.__name__, fit_score, test_score))

        file = open(cur_model_file, 'wb')
        pickle.dump(optimizedModel, file)
        file.close()
