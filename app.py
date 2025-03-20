from flask import Flask, render_template, request
from predictor import Predictor
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    prediction_date = request.form['date']
    start_date = request.form['start_date']
    end_date = request.form['end_date']
    if start_date > end_date:
        return render_template('prediction_result.html', prediction_result="Error: Start date cannot be after end date.")
    model = Predictor('AAPL', start_date, end_date)
    prediction = model.predict(prediction_date)
    prediction_result = (
        f"Stock price prediction for {prediction_date}  is {prediction[0][0]}"
    )
    return render_template('prediction_result.html', prediction_result=prediction_result)


def main():
    app.run(debug=True)
