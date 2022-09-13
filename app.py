from flask import Flask,render_template,url_for,request,redirect
import numpy as np
import pandas as pd
import joblib
import pickle


app = Flask(__name__)

model = joblib.load('model.pkl')


@app.route('/')
@app.route('/main')
def main():
	return render_template('main.html')

@app.route('/predict',methods=['POST'])
def predict():
	int_features=[[x for x in request.form.values()]]
	c = ['Avg. Area Income','Avg. Area House Age', 'Avg. Area Number of Rooms','Avg. Area Number of Bedrooms', 'Area Population']
	fin = pd.DataFrame(int_features,columns=c)
	result = model.predict(fin)
	return render_template("main.html",prediction_text=" The Estimated Hose Price is: {}".format(result))
	
	
    
    
	


if __name__ == "__main__":
	app.debug=True
	app.run(host = '127.0.0.1', port = 8000)