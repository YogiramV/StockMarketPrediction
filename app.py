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

        # Call the predict function from predictor.py to get the predictions
        forecast_data = predict(company, predict_date)

        # Format the data for display in HTML
        formatted_data = []
        for date, price in forecast_data:
            formatted_data.append({"date": date, "price": round(price, 4)})

        # Render the prediction result page and pass the forecast data
        formatted_date = datetime.strptime(
            predict_date, '%Y-%m-%d').strftime('%d-%m-%Y')
        return render_template('prediction.html',
                               company=company,
                               predict_date=formatted_date,
                               forecast_data=formatted_data)

    except Exception as e:
        return render_template('error.html', error_message=f"Error: {str(e)}. Please check your inputs.")


if __name__ == '__main__':
    app.run(debug=True)
