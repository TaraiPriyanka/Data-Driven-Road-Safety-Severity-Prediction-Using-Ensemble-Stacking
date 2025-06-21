from flask import Blueprint, render_template, request
from .utils import predict_accident_severity

main = Blueprint("main", __name__)

@main.route("/")
def index():
    return render_template("index.html")

@main.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        input_data = request.form.to_dict()
        prediction, processed_data = predict_accident_severity(input_data)
        return render_template("result.html", input_data=processed_data, prediction=prediction)
    return render_template("predict.html")
