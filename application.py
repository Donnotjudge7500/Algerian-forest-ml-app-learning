from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle


application = Flask(__name__)
app = application



# Importing the model
model_ridge = pickle.load(open('./models/ridge.pkl', 'rb'))

model_scale = pickle.load(open('./models/scaler.pkl', 'rb'))

# with open('./models/ridge.pkl', 'rb') as file:
#     model_ridge = pickle.load(file)



# Route for handling the home page
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict",methods = ['GET', 'POST'])
def predict():
    if request.method == 'POST':
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        transformed_data = model_scale.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
        
        model_prediction = model_ridge.predict(transformed_data)
        
        return render_template('predict.html', prediction_text = f'Predicted FWI: {model_prediction[0]}')
        
    else:
        return render_template('predict.html')


if __name__ == '__main__':
    app.run(host = "0.0.0.0") 