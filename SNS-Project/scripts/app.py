from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from datetime import datetime
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pickle

app = Flask(__name__)

# Load the trained models and Scalers
gru_dxy_model = load_model('gru_dxy_model.h5')

# Use the MinMaxScaler saved during training to avoid fitting issues
with open('scaler_dxy.pkl', 'rb') as f:
    scaler_dxy = pickle.load(f)

with open('scaler_gold.pkl', 'rb') as f:
    scaler_gold = pickle.load(f)

# Load the RandomForest and MLP models
with open('rf_gold_model.pkl', 'rb') as f:
    rf_gold_model = pickle.load(f)

with open('mlp_gold_model.pkl', 'rb') as f:
    mlp_gold_model = pickle.load(f)

# Function to predict the Dollar Index using the GRU model
def predict_dxy_gru(model, future_features):
    dxy_pred_scaled = model.predict(future_features)
    return scaler_dxy.inverse_transform(dxy_pred_scaled)  # Inverse transform to get the original scale

# Function to predict the Gold price
def predict_gold(future_features, predicted_dxy):
    rf_gold_pred = rf_gold_model.predict(future_features)
    mlp_gold_pred = mlp_gold_model.predict(future_features)
    gold_pred_scaled = (rf_gold_pred + mlp_gold_pred) / 2
    
    # Ensure that the prediction results are inverse-transformed
    gold_pred = scaler_gold.inverse_transform(gold_pred_scaled.reshape(-1, 1))
    
    return gold_pred[0][0]  # Return the inverse-transformed gold price prediction

# Logic to handle user request for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the user-inputted date from the frontend
        date_input = request.form['date_input']
        
        # Convert the date to datetime format
        future_date = datetime.strptime(date_input, "%Y-%m-%d")
        
        # Calculate date-related features
        future_day_of_week = future_date.weekday()
        future_month = future_date.month
        
        # Create future features (assuming lagged features as 0)
        future_features = pd.DataFrame([[0, 0, 0, 0, 0, 0, future_day_of_week, future_month]],
                                       columns=["DXY_Lag1", "DXY_Lag7", "DXY_Lag30", "DXY_Lag60", "DXY_MA30", "DXY_MA60", "day_of_week", "month"])
        
        future_features_gru = np.array(future_features).reshape((future_features.shape[0], 1, future_features.shape[1]))
        
        # Predict the Dollar Index
        future_dxy = predict_dxy_gru(gru_dxy_model, future_features_gru)
        
        # Predict the Gold price
        predicted_gold = predict_gold(future_features, future_dxy[0][0])
        
        # Convert the prediction results to standard Python float type to avoid JSON serialization errors
        result = {
            "dxy_predicted": float(future_dxy[0][0]),  # Convert to float type
            "gold_predicted": float(predicted_gold)    # Convert to float type
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)})

# Route to render the HTML page
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
