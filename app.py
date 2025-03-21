from flask import Flask, render_template, request
from predictor import predict
from datetime import datetime

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/forecast', methods=['POST'])
def forecast():
    try:
        # Get user input from the form
        company = request.form['company']
        predict_date = request.form['predict_date']

        # Call the predict function from predictor.py
        forecast = predict(company, predict_date)

        # Render the prediction result page and pass the forecast data
        formatted_date = datetime.strptime(
            predict_date, '%Y-%m-%d').strftime('%d-%m-%Y')
        return render_template('prediction.html',
                               company=company,
                               predict_date=formatted_date,
                               forecast=forecast)

    except Exception as e:
        return f"Error: {str(e)}", 400


if __name__ == '__main__':
    app.run(debug=True)
