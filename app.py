from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import pickle
import os
from datetime import datetime, timedelta
import random

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'

# Database setup
def init_db():
    conn = sqlite3.connect('flight_predictor.db')
    c = conn.cursor()
    
    # Users table
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  email TEXT UNIQUE NOT NULL,
                  password TEXT NOT NULL)''')
    
    # Flight searches table
    c.execute('''CREATE TABLE IF NOT EXISTS searches
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  departure TEXT,
                  destination TEXT,
                  travel_date TEXT,
                  predicted_price REAL,
                  search_date TEXT,
                  FOREIGN KEY (user_id) REFERENCES users (id))''')
    
    conn.commit()
    conn.close()

# Generate sample training data and train model
def create_sample_model():
    # Create sample flight data
    np.random.seed(42)
    
    airlines = ['IndiGo', 'SpiceJet', 'Air India', 'Vistara', 'GoAir']
    cities = ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata', 'Hyderabad', 'Pune', 'Ahmedabad']
    
    data = []
    for _ in range(1000):
        airline = random.choice(airlines)
        source = random.choice(cities)
        destination = random.choice([c for c in cities if c != source])
        duration = random.randint(60, 300)  # minutes
        days_left = random.randint(1, 90)
        stops = random.choice([0, 1, 2])
        
        # Price calculation with some logic
        base_price = 3000
        if airline == 'Vistara': base_price += 1000
        if airline == 'Air India': base_price += 500
        if stops == 0: base_price += 800
        if days_left < 7: base_price += 1500
        elif days_left < 30: base_price += 500
        if duration > 180: base_price += 300
        
        price = base_price + random.randint(-800, 1200)
        
        data.append({
            'Airline': airline,
            'Source': source,
            'Destination': destination,
            'Duration': duration,
            'Days_Left': days_left,
            'Stops': stops,
            'Price': price
        })
    
    df = pd.DataFrame(data)
    
    # Encode categorical variables
    le_airline = LabelEncoder()
    le_source = LabelEncoder()
    le_dest = LabelEncoder()
    
    df['Airline_encoded'] = le_airline.fit_transform(df['Airline'])
    df['Source_encoded'] = le_source.fit_transform(df['Source'])
    df['Destination_encoded'] = le_dest.fit_transform(df['Destination'])
    
    # Features and target
    X = df[['Airline_encoded', 'Source_encoded', 'Destination_encoded', 'Duration', 'Days_Left', 'Stops']]
    y = df['Price']
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Save model and encoders
    with open('flight_model.pkl', 'wb') as f:
        pickle.dump({
            'model': model,
            'le_airline': le_airline,
            'le_source': le_source,
            'le_dest': le_dest,
            'airlines': airlines,
            'cities': cities
        }, f)

# Load or create model
def load_model():
    if not os.path.exists('flight_model.pkl'):
        create_sample_model()
    
    with open('flight_model.pkl', 'rb') as f:
        return pickle.load(f)

model_data = load_model()

@app.route('/')
def index():
    if 'user_id' in session:
        return render_template('dashboard.html', username=session['username'])
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        if not username or not email or not password:
            flash('All fields are required!')
            return render_template('register.html')
        
        conn = sqlite3.connect('flight_predictor.db')
        c = conn.cursor()
        
        try:
            hashed_password = generate_password_hash(password)
            c.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                     (username, email, hashed_password))
            conn.commit()
            flash('Registration successful! Please login.')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username or email already exists!')
        finally:
            conn.close()
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = sqlite3.connect('flight_predictor.db')
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = c.fetchone()
        conn.close()
        
        if user and check_password_hash(user[3], password):
            session['user_id'] = user[0]
            session['username'] = user[1]
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password!')
    
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html', 
                         username=session['username'],
                         airlines=model_data['airlines'],
                         cities=model_data['cities'])

@app.route('/predict', methods=['POST'])
def predict():
    if 'user_id' not in session:
        return jsonify({'error': 'Please login first'})
    
    try:
        data = request.json
        airline = data['airline']
        source = data['source']
        destination = data['destination']
        travel_date = data['travel_date']
        duration = int(data['duration'])
        stops = int(data['stops'])
        
        # Calculate days left
        travel_dt = datetime.strptime(travel_date, '%Y-%m-%d')
        days_left = (travel_dt - datetime.now()).days
        
        if days_left < 0:
            return jsonify({'error': 'Travel date must be in the future'})
        
        # Encode inputs
        airline_encoded = model_data['le_airline'].transform([airline])[0]
        source_encoded = model_data['le_source'].transform([source])[0]
        dest_encoded = model_data['le_dest'].transform([destination])[0]
        
        # Predict
        features = np.array([[airline_encoded, source_encoded, dest_encoded, duration, days_left, stops]])
        predicted_price = model_data['model'].predict(features)[0]
        
        # Round to nearest 50
        predicted_price = round(predicted_price / 50) * 50
        
        # Save search history
        conn = sqlite3.connect('flight_predictor.db')
        c = conn.cursor()
        c.execute("""INSERT INTO searches 
                     (user_id, departure, destination, travel_date, predicted_price, search_date)
                     VALUES (?, ?, ?, ?, ?, ?)""",
                 (session['user_id'], source, destination, travel_date, 
                  predicted_price, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        conn.commit()
        conn.close()
        
        return jsonify({
            'predicted_price': int(predicted_price),
            'message': 'Price predicted successfully!'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/history')
def history():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    conn = sqlite3.connect('flight_predictor.db')
    c = conn.cursor()
    c.execute("""SELECT departure, destination, travel_date, predicted_price, search_date
                 FROM searches WHERE user_id = ? ORDER BY search_date DESC LIMIT 10""",
             (session['user_id'],))
    searches = c.fetchall()
    conn.close()
    
    return render_template('history.html', searches=searches)

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out successfully!')
    return redirect(url_for('index'))

if __name__ == '__main__':
    init_db()
    app.run(debug=True)