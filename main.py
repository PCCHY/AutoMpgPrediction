from flask import Flask, request, jsonify
from autompg_model_files.ml_model import predict_mpg
import pickle

app = Flask('Auto MPG service')

@app.route('/test', methods = ['GET'])
def test():
    return 'Pinging Model Application!!'

@app.route('/predict', methods = ['POST'])
def predict():
    vehicle = request.get_json()
    with open('./autompg_model_files/model.bin','rb') as f_in:
        model = pickle.load(f_in)

    predictions = predict_mpg(vehicle, model)

    result = {
        'mpg_prediction':list(predictions)
        }

    return jsonify(result)
    

if __name__ == '__main__':
    app.run(debug = True, host='0.0.0.0', port=8000)
