from flask import Flask, render_template, redirect, url_for, request,jsonify


import numpy as np

import pickle

app=Flask(__name__)

f=open('ada_model.pkl','rb')
ml_model=pickle.load(f)


@app.route("/")
def index():
    return render_template("index.html")
    
@app.route('/view_report')
def view_report():
    return redirect(url_for('static', filename='data_report.html'))

@app.route("/predict",methods=["POST","GET"])
def predict():
        if request.method=="POST":
                v1 = int(request.form.get('Gender'))
                v2 = float(request.form.get('Age'))
                #v3= int(request.form.get('Hypertension'))
                #v4 =int(request.form.get('HeartDisease'))
                v5 =int(request.form.get('Smokinghistory'))
                v6 = float(request.form.get('Bmi'))
                v7= float(request.form.get('Hba1c'))
                v8 =float(request.form.get('Bloodglucose'))
                mysample = np.array([v1,v2,v5,v6,v7,v8])
                ex1 = mysample.reshape(1,-1)
                ypred=ml_model.predict(ex1)
                output=ypred[0]
                
                if output==0:
                        ans="No Diabetes"
                else:
                        ans="Diabetes"
                
                return render_template("index.html",prediction_text=ans)


if __name__=="__main__":
    app.run()
