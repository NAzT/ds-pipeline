import os
import pickle


# The class used for retrieving static resources
class GetResources:
    # loop through the model folder, read the first model in pickle format, and store them in the model variable  
    @staticmethod
    def getModel():
        for filename in os.listdir("model"):
            if filename.endswith(".pickle"):
                cur_model_file = os.path.join('model', filename)
                file = open(cur_model_file, 'rb')
                model = pickle.load(file)
                file.close()
                break
        return model
