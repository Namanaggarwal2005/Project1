from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load trained model
model = pickle.load(open("house_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

def yes_no(val):
    return 1 if val == "yes" else 0


@app.route("/predict", methods=["POST"])
def predict():
    area = float(request.form["area"])
    bedrooms = int(request.form["bedrooms"])
    bathrooms = int(request.form["bathrooms"])
    stories = int(request.form["stories"])
    parking = int(request.form["parking"])

    mainroad = yes_no(request.form["mainroad"])
    guestroom = yes_no(request.form["guestroom"])
    basement = yes_no(request.form["basement"])
    hotwaterheating = yes_no(request.form["hotwaterheating"])
    airconditioning = yes_no(request.form["airconditioning"])
    prefarea = yes_no(request.form["prefarea"])

    furnishingstatus = request.form["furnishingstatus"]

    input_data = pd.DataFrame([{
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "stories": stories,
        "parking": parking,
        "mainroad": mainroad,
        "guestroom": guestroom,
        "basement": basement,
        "hotwaterheating": hotwaterheating,
        "airconditioning": airconditioning,
        "prefarea": prefarea,
        "furnishingstatus": furnishingstatus
    }])

    prediction = model.predict(input_data)[0]

    return render_template(
        "index.html",
        prediction_text=f"₹ {round(prediction, 2)}"
    )



if __name__ == "__main__":
    app.run(debug=True)

