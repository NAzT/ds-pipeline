from createBlindTestSamples import CreateBlindTestSamples
from createModels import CreateModels
import numpy as np
from sklearn import *


def get_model(ith_technique):
    if ith_technique == 0:
        return linear_model.LinearRegression()
    elif ith_technique == 1:
        return linear_model.SGDRegressor(max_iter=1e4, tol=1e-3)
    elif ith_technique == 2:
        return linear_model.ElasticNet()
    elif ith_technique == 3:
        return svm.SVR(kernel='rbf')
    elif ith_technique == 4:
        return ensemble.RandomForestRegressor()
    elif ith_technique == 5:
        return ensemble.GradientBoostingRegressor()
    elif ith_technique == 6:
        return neighbors.KNeighborsRegressor()


def get_param(ith_technique):
    if ith_technique == 0:
        return {'model__normalize': [True, False]}
    elif ith_technique == 1:
        return {'model__alpha': [0.01, 0.001, 0.0001], 'model__penalty': ['l1', 'l2'],
                'model__loss': ['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']}
    elif ith_technique == 2:
        return {'model__alpha': np.r_[0.01:1:0.05], 'model__normalize': [True, False]}
    elif ith_technique == 3:
        return {'model__C': [1, 10, 1e2, 1e3], 'model__gamma': [1e-3, 1e-2, 0.1, 1, 10, 1e2, 1e3]}
    elif ith_technique == 4:
        return {'model__min_samples_split': [2, 4, 8, 16]}
    elif ith_technique == 5:
        return {'model__learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3], 'model__alpha': [0.01, 0.1, 1 - (1e-2)],
                'model__min_samples_split': [2, 4, 8, 16]}
    elif ith_technique == 6:
        return {'model__n_neighbors': [1, 3, 5], 'model__metric': ['manhattan', 'euclidean']}


import warnings

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    CreateBlindTestSamples.create()
    N_technique = 7
    for i in range(0, N_technique):
        model = get_model(i)
        param = get_param(i)
        CreateModels.create(model, param)
