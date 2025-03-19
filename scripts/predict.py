import requests
import json
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
import os

def fetch_penguin_data():
    """Fetch new penguin data from the API"""
    api_url = "http://130.225.39.127:8000/new_penguin/"
    response = requests.get(api_url)
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"API request failed with status code {response.status_code}")

def load_model_components():
    """Load the trained model and preprocessors"""
    clf = joblib.load('../models/penguin_classifier.joblib')
    scaler = joblib.load('../models/penguin_scaler.joblib')
    return clf, scaler

def make_prediction(data, clf, scaler):
    """Preprocess the data and make a prediction"""
    
    features = pd.DataFrame({
        'bill_depth_mm': [data['bill_depth_mm']],
        'flipper_length_mm': [data['flipper_length_mm']]
    })
    
    # Scale numerical features
    features_scaled = scaler.transform(features)
    
    prediction = clf.predict(features_scaled)
    
    return prediction[0]

def update_prediction_history(new_prediction, new_data):
    """Update the prediction history JSON file"""
    history_file = '../data/predictions.json'
    
    # Load existing history or create new
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            history = json.load(f)
    else:
        history = []
    
    # Add new prediction
    prediction_entry = {
        'prediction': new_prediction,
        'bill_length_mm': new_data['bill_length_mm'],
        'bill_depth_mm': new_data['bill_depth_mm'],
        'flipper_length_mm': new_data['flipper_length_mm'],
        'body_mass_g': new_data['body_mass_g']
    }
    
    history.append(prediction_entry)
    
    # Save updated history
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)
    
    return history

def update_html_page(history):
    """Update the GitHub Pages HTML file with the latest predictions"""
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Daily Penguin Predictions</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {{ padding: 20px; }}
        .prediction-card {{ margin-bottom: 15px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1 class="mt-4 mb-4">Daily Penguin Species Predictions</h1>
        <p class="lead">Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC</p>
        
        <h2 class="mt-4">Latest Prediction</h2>
        <div class="card prediction-card">
            <div class="card-body">
                <h6 class="card-subtitle mb-2 text-muted">Predicted Species: {history[-1]['prediction']}</h6>
                <p class="card-text">
                    <strong>Measurements:</strong><br>
                    Bill Length: {history[-1]['bill_length_mm']:.2f} mm<br>
                    Bill Depth: {history[-1]['bill_depth_mm']:.2f} mm<br>
                    Flipper Length: {history[-1]['flipper_length_mm']:.2f} mm<br>
                    Body Mass: {history[-1]['body_mass_g']:.2f} g
                </p>
            </div>
        </div>
        
        <h2 class="mt-4">Prediction History</h2>
        <div class="table-responsive">
            <table class="table table-striped">
                <thead>
                    <tr>
                        <th>Predicted Species</th>
                        <th>Bill Length (mm)</th>
                        <th>Bill Depth (mm)</th>
                        <th>Flipper Length (mm)</th>
                        <th>Body Mass (g)</th>
                    </tr>
                </thead>
                <tbody>
"""
    
    # Add table rows for each prediction (most recent first)
    for entry in reversed(history):
        html_content += f"""
                    <tr>
                        <td>{entry['prediction']}</td>
                        <td>{entry['bill_length_mm']:.2f}</td>
                        <td>{entry['bill_depth_mm']:.2f}</td>
                        <td>{entry['flipper_length_mm']:.2f}</td>
                        <td>{entry['body_mass_g']:.2f}</td>
                    </tr>
"""
    
    html_content += """
                </tbody>
            </table>
        </div>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""
    
    with open('../index.html', 'w') as f:
        f.write(html_content)

def main():
    print(os.getcwd())
    # Fetch new data
    print("Fetching new penguin data...")
    new_data = fetch_penguin_data()
    print(f"Received data: {new_data}")
    
    # Load model components
    print("Loading model components...")
    clf, scaler = load_model_components()
    
    # Make prediction
    print("Making prediction...")
    prediction = make_prediction(new_data, clf, scaler)
    print(f"Predicted species: {prediction}")
    
    # Update prediction history
    print("Updating prediction history...")
    history = update_prediction_history(prediction, new_data)
    
    # Update HTML page
    print("Updating HTML page...")
    update_html_page(history)
    
    print("Done!")

if __name__ == "__main__":
    main()
