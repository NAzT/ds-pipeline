import pandas as pd
from flask import Flask, jsonify, request
from estimationService import GetResources
from collections import OrderedDict

app = Flask(__name__)
app.model = GetResources.getModel()


@app.route('/predict', methods=['GET'])
def predict():
    # Store the query-string argument in the arg_list variable and then convert its format to pandas dataframe
    arg_list = request.args.to_dict(flat=False)
    query_df = pd.DataFrame.from_dict(OrderedDict(arg_list))
    print(arg_list)
    for feat in query_df.columns:
        if isinstance(query_df[feat], object):
            query_df[feat] = query_df[feat].str.replace(",", "").astype("float64")
    else:
        query_df[feat] = query_df[feat].astype("float64")
    # For each model, generated the estimated value.
    try:
        print(query_df)
        predict_val = app.model.predict(query_df).astype('float64')[0]
        return jsonify(predict=str(predict_val))
    except Exception as ex:
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print(message)
        return "Some input parameters are missing " + message


if __name__ == '__main__':
    app.run()
