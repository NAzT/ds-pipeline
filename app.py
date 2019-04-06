from createBlindTestSamples import CreateBlindTestSamples
from createModels import CreateModels
from sklearn import linear_model

if __name__ == '__main__':
    CreateBlindTestSamples.create()
    model = linear_model.LinearRegression()
    CreateModels.create(model)
