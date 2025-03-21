import os
import sys
import webbrowser
import threading
import time
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from datetime import datetime
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pickle

# Define the Flask app
app = Flask(__name__)

# Load pre-trained models and scalers
def load_models_and_scalers():
    global gru_dxy_model, scaler_dxy, scaler_gold, rf_gold_model, mlp_gold_model
    
    # Get the directory of the executable or script
    if getattr(sys, 'frozen', False):
        # If running as a bundled executable
        base_dir = os.path.dirname(sys.executable)
    else:
        # If running as a script
        base_dir = os.path.dirname(os.path.abspath(__file__))
    
    try:
        # Load the GRU model for Dollar Index prediction
        gru_dxy_model = load_model(os.path.join(base_dir, 'gru_dxy_model.h5'))
        
        # Load the MinMaxScaler for Dollar Index
        with open(os.path.join(base_dir, 'scaler_dxy.pkl'), 'rb') as f:
            scaler_dxy = pickle.load(f)
        
        # Load the MinMaxScaler for Gold
        with open(os.path.join(base_dir, 'scaler_gold.pkl'), 'rb') as f:
            scaler_gold = pickle.load(f)
        
        # Load the RandomForest model for Gold prediction
        with open(os.path.join(base_dir, 'rf_gold_model.pkl'), 'rb') as f:
            rf_gold_model = pickle.load(f)
        
        # Load the MLP model for Gold prediction
        with open(os.path.join(base_dir, 'mlp_gold_model.pkl'), 'rb') as f:
            mlp_gold_model = pickle.load(f)
            
        return True
    except Exception as e:
        print(f"Error loading models: {e}")
        return False

# Function to predict Dollar Index using GRU model
def predict_dxy_gru(model, future_features):
    dxy_pred_scaled = model.predict(future_features)
    return scaler_dxy.inverse_transform(dxy_pred_scaled)

# Function to predict Gold price
def predict_gold(future_features, predicted_dxy):
    rf_gold_pred = rf_gold_model.predict(future_features)
    mlp_gold_pred = mlp_gold_model.predict(future_features)
    gold_pred_scaled = (rf_gold_pred + mlp_gold_pred) / 2
    
    # Ensure that prediction results are inverse-scaled
    gold_pred = scaler_gold.inverse_transform(gold_pred_scaled.reshape(-1, 1))
    
    return gold_pred[0][0]

# Route to handle prediction requests
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the date input from the frontend
        date_input = request.form['date_input']
        
        # Convert date to datetime format
        future_date = datetime.strptime(date_input, "%Y-%m-%d")
        
        # Calculate date-related features
        future_day_of_week = future_date.weekday()
        future_month = future_date.month
        
        # Create future features (assuming lagged features as 0)
        future_features = pd.DataFrame([[0, 0, 0, 0, 0, 0, future_day_of_week, future_month]],
                                      columns=["DXY_Lag1", "DXY_Lag7", "DXY_Lag30", "DXY_Lag60", "DXY_MA30", "DXY_MA60", "day_of_week", "month"])
        
        future_features_gru = np.array(future_features).reshape((future_features.shape[0], 1, future_features.shape[1]))
        
        # Predict Dollar Index
        future_dxy = predict_dxy_gru(gru_dxy_model, future_features_gru)
        
        # Predict Gold price
        predicted_gold = predict_gold(future_features, future_dxy[0][0])
        
        # Convert prediction results to standard Python float type to avoid JSON serialization errors
        result = {
            "dxy_predicted": float(future_dxy[0][0]),
            "gold_predicted": float(predicted_gold)
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)})

# Route to render the HTML page
@app.route('/')
def index():
    return render_template('index.html')

def open_browser():
    """Function to open the browser after a short delay"""
    time.sleep(2)  # Give the server time to start
    webbrowser.open('http://127.0.0.1:5000/')

if __name__ == '__main__':
    # Load models and scalers
    if load_models_and_scalers():
        # Start browser in a separate thread
        threading.Thread(target=open_browser).start()
        
        # Run the Flask app
        app.run(debug=False)
    else:
        print("Failed to load models. Please ensure all model files exist in the correct location.")
        input("Press Enter to exit...") 