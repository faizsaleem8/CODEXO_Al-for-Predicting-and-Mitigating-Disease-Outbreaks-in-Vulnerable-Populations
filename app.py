from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import pandas as pd
import numpy as np
import requests  # Ensure this is imported

# Add your OpenWeatherMap API Key here
OPEN_WEATHER_MAP_API_KEY = '2c84a58594dea5397b1a0b140de2ebcf'

app = Flask(__name__)
app.secret_key = 'Hello1234'  # Replace with a strong secret key

# Load CSV files
climate_data = pd.read_csv('climate_data_monthly.csv')
disease_data = pd.read_csv('disease_data_monthly.csv')
healthcare_data = pd.read_csv('healthcare_data.csv')
population_data = pd.read_csv('population_data.csv')
district_coordinates = pd.read_csv('district_coordinates.csv') 

# Simulate a simple user database
users = {'admin': '123', 'user': 'user123'}  

# Get unique states and districts for dropdown
states = sorted(climate_data['State'].unique().tolist())
district_mapping = {state: sorted(climate_data[climate_data['State'] == state]['District'].unique().tolist()) for state in states}
@app.route('/')
def login():
    return render_template('login.html')  # Login page

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contacts')
def contacts():
    return render_template('contacts.html')

@app.route('/docs')
def docs():
    return render_template('docs.html')

@app.route('/login', methods=['POST'])
def do_login():
    username = request.form.get('username')
    password = request.form.get('password')

    # Validate login
    if username in users and users[username] == password:
        session['user'] = username  # Set session for logged-in user
        return redirect(url_for('index'))  # Redirect to index page
    else:
        return render_template('login.html', error='Invalid username or password')

# Define the function to add users to the dictionary
def add_user(username, password):
    users[username] = password

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        # Check if passwords match
        if password != confirm_password:
            return render_template('signup.html', error="Passwords do not match.")

        # Check if the username already exists
        if username in users:
            return render_template('signup.html', error="Username already exists.")

        # Call the add_user function
        add_user(username, password)
        return redirect(url_for('login'))

    return render_template('signup.html')


@app.route('/logout')
def logout():
    session.pop('user', None)  # Clear user session
    return redirect(url_for('login'))  # Redirect to login page

@app.route('/index')
def index():
    if 'user' not in session:
        return redirect(url_for('login'))  # Redirect to login if not authenticated
    return render_template('index.html', states=states)

@app.route('/get_districts', methods=['POST'])
def get_districts():
    state = request.form.get('state')
    if state:
        districts = district_mapping.get(state, [])
        return jsonify({'districts': districts})
    return jsonify({'districts': []})

@app.route('/get_weather', methods=['POST'])
def get_weather():
    state = request.form.get('state')
    district = request.form.get('district')
    if state and district:
        # Use OpenWeatherMap API to get current weather
        response = requests.get(
            f'http://api.openweathermap.org/data/2.5/weather',
            params={
                'q': f'{district},{state},India',
                'appid': OPEN_WEATHER_MAP_API_KEY,
                'units': 'metric'  # To get temperature in Celsius
            }
        )
        if response.status_code == 200:
            data = response.json()
            return jsonify({
                'temperature': data['main']['temp'],
                'humidity': data['main']['humidity'],
                'rainfall': data.get('rain', {}).get('1h', 0)  # Rainfall in last 1 hour if available
            })
        else:
            return jsonify({'error': 'Unable to fetch weather data'})
    return jsonify({'error': 'Weather data not found'})

disease_info = pd.read_csv('disease_symptoms_precautions_with_cure.csv')
@app.route('/predict', methods=['POST'])
def predict():
    state = request.form.get('state')
    district = request.form.get('district')

    if state and district:
        # Filter disease data for the specific district
        district_data = disease_data[
            (disease_data['State'] == state) & (disease_data['District'] == district)
        ]

        if not district_data.empty:
            # Find the disease with the most reported cases
            top_disease = district_data.groupby('Disease')['ReportedCases'].sum().idxmax()
            top_disease_details = disease_info[disease_info['Disease'] == top_disease].iloc[0]

            # Simulated outbreak prediction logic
            outbreak_probability = np.random.uniform(0.5, 0.9)  # Randomized confidence
            if outbreak_probability > 0.8: outbreak = 'High'
            elif (outbreak_probability <= 0.8 and outbreak_probability >= 0.6) : 'Moderate'
            else: outbreak = 'Low'

            # Get hospital count for the district
            hospitals_count = healthcare_data[
                (healthcare_data['State'] == state) & (healthcare_data['District'] == district)
            ]['Hospitals'].values[0]

            # Prepare the prediction message
            prediction_message = (
                f"<strong>{outbreak} outbreak risk.</strong><br>"
                f"<strong>Number of hospitals:</strong> {hospitals_count}<br><br>"
                f"<strong>Disease with the most reported cases:</strong> {top_disease}<br>"
                f"<strong>Symptoms:</strong><br>"
                f"&bull; {top_disease_details['Symptoms'].replace(',', '<br>&bull; ')}<br>"
                f"<strong>Precautions:</strong><br>"
                f"&bull; {top_disease_details['Precautions'].replace(',', '<br>&bull; ')}<br>"
                f"<strong>Cure:</strong><br>"
                f"&bull; {top_disease_details['Cure'].replace(',', '<br>&bull; ')}"
            )

            return jsonify({
                'prediction': prediction_message,
                'ai_confidence': f"{outbreak_probability * 100:.2f}%"
            })

        return jsonify({'error': 'No data available for the selected district'})

    return jsonify({'error': 'State and district are required'})


@app.route('/heatmap_data', methods=['GET'])
def heatmap_data():
    heatmap_points = []
    for index, row in district_coordinates.iterrows():
        district = row['District']
        lat = row['Latitude']
        lon = row['Longitude']

        # Simulate outbreak prediction for heatmap (replace with actual logic)
        outbreak_probability = np.random.uniform(0.5, 0.9)  # Random confidence value for demonstration

        heatmap_points.append([lat, lon, outbreak_probability])  # [latitude, longitude, intensity]

    return jsonify(heatmap_points)

if __name__ == '__main__':
    app.run(debug=True,port=8800)