from flask import Flask, request, jsonify
from joblib import load
from flask_cors import CORS
import pandas as pd

scalar = load("scalar.joblib")
model = load("chronicKidney.joblib")

app = Flask(__name__)
CORS(app)

@app.route('/')
def hello():
    return 'Hello World!'

@app.route('/chronic-kidney', methods=['POST'])
def kidney():
    try:
        data = request.get_json()
        data_val=[list(data.values())]

        dataframe = pd.DataFrame(data_val)
        print(dataframe)
        dataframe.iloc[:, [0,2,4,6,7]] = scalar.transform(dataframe.iloc[:, [0,2,4,6,7]])

        predict = model.predict(dataframe)[0]
        print(predict)

        return jsonify({'class' : str(predict)})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
