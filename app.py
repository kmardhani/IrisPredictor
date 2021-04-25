import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('Iris-logistic', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    sepal_length = float(request.form['sl'])
    sepal_width = float(request.form['sw'])
    petal_length = float(request.form['pl'])
    petal_width = float(request.form['pw'])
    features = np.array([[sepal_length,sepal_width,petal_length,petal_width]])
    species = model.predict(features)

    return render_template('index.html', prediction_text='Predicted flower species is {}'.format(species[0]))


if __name__ == "__main__":
    app.run(debug=True)
