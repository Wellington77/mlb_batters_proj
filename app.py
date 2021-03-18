# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 01:44:15 2021

@author: welli
"""

from flask import Flask, render_template, request
import pickle
import numpy as np


app = Flask(__name__)

model = pickle.load(open('model.pkl' , 'rb'))


#printing model
#print(model.predict([[162,170,26,500,457]]).reshape(1,-1))


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
     data1= request.form['a']
     data2= request.form['b']
     data3= request.form['c']
     data4= request.form['d']
     data5= request.form['e']
     arr = np.array([[data1, data2, data3, data4, data5]],dtype='float64')
     pred = model.predict(arr)

     return render_template('home.html', prediction_text = "your batting average should be around {}".format(pred))

# =============================================================================
#     int_features = [int(x) for x in request.form.values()]
#     final_features = [np.array(int_features)]
#     pred = model.predict(final_features)
# =============================================================================
    


if __name__ == "__main__":
    app.run(debug = True)
    
