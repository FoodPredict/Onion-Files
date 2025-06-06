# app.py

import os
import joblib
import pandas as pd
from flask import Flask, request, render_template_string
import pyngrok # While not needed for Render deployment directly, keep it if you want to test locally later
import threading
import time

# --- Configuration ---
MODEL_FILE_NAME = 'best_rf_model.joblib'
INDEX_HTML_FILE_NAME = 'index.html'
FLASK_PORT = int(os.environ.get('PORT', 5000)) # Get port from environment variable, default to 5000

# --- Load the Model ---
# Construct the path relative to where app.py is located
# Assuming the model file is in the same directory as app.py for simplicity initially
model_path = os.path.join(os.path.dirname(__file__), MODEL_FILE_NAME)

try:
    loaded_model = joblib.load(model_path)
    print(f"Model successfully loaded from: {model_path}")
except FileNotFoundError:
    print(f"Error: Model file not found at {model_path}.")
    loaded_model = None
except Exception as e:
    print(f"An error occurred while loading the model: {e}")
    loaded_model = None

# --- Load the HTML Template ---
# Construct the path to the index.html file
# Assuming a 'templates' folder in the same directory as app.py
templates_folder = os.path.join(os.path.dirname(__file__), 'templates')
index_html_path = os.path.join(templates_folder, INDEX_HTML_FILE_NAME)

try:
    with open(index_html_path, 'r') as f:
        index_html_content = f.read()
    print(f"HTML template loaded from: {index_html_path}")
except FileNotFoundError:
    print(f"Error: {INDEX_HTML_FILE_NAME} not found at {index_html_path}.")
    index_html_content = "<html><body><h1>Error loading template.</h1></body></html>"
except Exception as e:
    print(f"An error occurred while loading the HTML template: {e}")
    index_html_content = "<html><body><h1>Error loading template.</h1></body></html>"

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Routes ---

@app.route('/')
def index():
    print("Index route accessed.", flush=True)
    return render_template_string(index_html_content)

@app.route('/predict', methods=['POST'])
def predict():
    print("Predict route accessed.", flush=True)
    if loaded_model is None:
        print("Error: Model not loaded in predict route.", flush=True)
        return "Error: Model not loaded.", 500

    try:
        storage_temperature = float(request.form['storage_temperature'])
        storage_duration = int(request.form['storage_duration'])
        texture = request.form['texture']
        microbial_load = request.form['microbial_load']
        weight_loss = float(request.form['weight_loss'])

        # --- Preprocessing (IMPLEMENT YOUR LOGIC HERE) ---
        try:
            texture_processed = float(texture)
        except ValueError:
            print(f"Warning: Could not convert texture '{texture}' to float. Using 0.0.", flush=True)
            texture_processed = 0.0

        try:
            microbial_load_processed = float(microbial_load)
        except ValueError:
            print(f"Warning: Could not convert microbial_load '{microbial_load}' to float. Using 0.0.", flush=True)
            microbial_load_processed = 0.0
        # *********************************************************

        input_data = pd.DataFrame([[storage_temperature, storage_duration, texture_processed, microbial_load_processed, weight_loss]],
                                  columns=['Storage Temperature', 'Storage Duration', 'Texture', 'Microbial Load', 'Weight Loss'])

        prediction = loaded_model.predict(input_data)[0]

        print(f"Prediction successful: {prediction}", flush=True)
        return f"Predicted Shelf Life: {prediction:.2f} days"

    except ValueError as ve:
        print(f"ValueError in predict route: {ve}", flush=True)
        return f"Invalid input: {ve}. Please ensure all fields are filled correctly.", 400
    except Exception as e:
        print(f"An error occurred during prediction: {e}", flush=True)
        return f"An error occurred during prediction: {e}", 500

if __name__ == '__main__':
    print(f"App is designed to run on port {FLASK_PORT}", flush=True)
    app.run(port=FLASK_PORT, host='0.0.0.0')