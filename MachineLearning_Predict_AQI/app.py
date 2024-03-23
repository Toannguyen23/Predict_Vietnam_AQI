import numpy as np
from flask import Flask, request, render_template
import pickle
#tao app
app = Flask(__name__, template_folder= 'template')

#tai model
model = pickle.load(open("air-quality-predict_1.pkl", "rb"))

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods = ["POST"])
def predict():
    if request.method == 'POST':
        hud = float(request.form['Humidity'])
        co = float(request.form['CO'])
        no2 = float(request.form['NO2'])
        o3 = float(request.form['O3'])
        pm10 = float(request.form['PM10'])
        pm_2 = float(request.form['PM2.5'])
        so2= float(request.form['SO2'])
        temp = float(request.form['Temperature'])
        wind = float(request.form['Wind'])

        data = np.array([[hud, co, no2, o3, pm10, pm_2,so2, temp, wind ]])
        my_prediction =model.predict(data)
        #them lenh mới
        if(my_prediction == 0 and my_prediction <= 50 ):
            result = "Chất lượng không khí TỐT "
        elif(my_prediction > 50 and my_prediction <= 100 ):
            result = "Chất lượng không khí TRUNG BÌNH "
        elif(my_prediction > 100 and my_prediction <= 150 ):
            result = "Chất lượng không khí KÉM"
        elif(my_prediction > 150 and my_prediction <= 200 ):
            result = "Chất lượng không khí XẤU"
        elif(my_prediction > 200 and my_prediction <= 300 ):
            result = "Chất lượng không khí RẤT XẤU"
        elif(my_prediction > 300 ):
            result = "Chất lượng không khí NGUY HẠI"
        #hang them lẹnh mơi
        return render_template('index.html', prediction_text = "Chỉ số không khí hiện tại: {}".format(my_prediction), result_predict = result)
if __name__ == "__main__":
    app.run(debug=True)