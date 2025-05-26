from flask import Flask,render_template,request
import joblib
import numpy as np

app = Flask(__name__)




@app.route('/')
def home():
    return render_template('home.html')
model = joblib.load("model.pkl")
@app.route("/predict", methods=["POST"])
def predict():
    age = int(request.form["age"])
    sex = int(request.form["sex"])
    cp = int(request.form["cp"])
    trestbps = int(request.form["trestbps"])
    chol = int(request.form["chol"])
    fbs = int(request.form["fbs"])
    thalach = int(request.form["thalach"])
    restecg=int(request.form["restecg"])
    exang=int(request.form["exang"])
    oldpeak=int(request.form["oldpeak"])
    slope=int(request.form["slope"])
    ca=int(request.form["ca"])
    condition=int(request.form["condition"])

    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, thalach,restecg,exang,oldpeak,slope,ca,condition]])
    prediction = model.predict(input_data)[0]

    result = "Wysokie ryzyko" if prediction == 1 else "Niskie ryzyko"
    return render_template("result.html", result=result)

@app.route('/result')
def result():
    return render_template('result.html')
if __name__ == '__main__':
    app.run()
