from operator import methodcaller
from flask import Flask,request,render_template,jsonify
from flask.templating import render_template_string
from flask_cors import CORS,cross_origin
import pandas as pd
import numpy as np
import pickle

app = Flask("__name__")
@app.route('/',methods=['GET'])
@cross_origin()
def main():
    return render_template('index.html')

@app.route('/predict',methods=['POST','GET'])
@cross_origin()
def predict():
    if request.method == 'POST':
        try:
            pclass = int(request.form['pclass'])
            sex = int(request.form['sex'])
            age = int(request.form['age'])
            sib = int(request.form['sib'])
            parch = int(request.form['parch'])
            fare = float(request.form['fare'])

            file = 'model.pickle'
            model = pickle.load(open(file,'rb'))
            predict = model.predict([[pclass,sex,age,sib,parch,fare]])
            result = ''
            pic=''
            if predict == 1:
                result='Survived'
                pic='sur.jpg'
            else:
                result= 'Not Survived'
                pic='rip.png'

            return render_template('result.html',result = result ,pic=pic)
        except Exception as e:
            print(e)
            return "SOMETHING WENT WRONG"
    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)