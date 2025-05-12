from flask import Flask, render_template, request
import requests
import os

app = Flask(__name__)
FASTAPI_URL = os.environ.get("FASTAPI_SERVER_URL", "http://fastapi:8000")

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            features = {
                'hour': int(request.form['hour']),
                'day_of_week': int(request.form['day_of_week']),
                'temp': float(request.form['temp']),
                'precip': float(request.form['precip']),
                'flight_arrivals': int(request.form['flight_arrivals'])
            }
            resp = requests.post(f"{FASTAPI_URL}/predict", json=features)
            resp.raise_for_status()
            prediction = resp.json()
        except Exception as e:
            prediction = {'error': str(e)}
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
