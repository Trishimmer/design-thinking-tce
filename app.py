from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

INPUT_LIMITS = {
    'nitrogen': (0, 140, 'Nitrogen'),
    'phosphorus': (0, 145, 'Phosphorus'),
    'potassium': (0, 205, 'Potassium'),
    'temperature': (0, 50, 'Temperature'),
    'humidity': (0, 100, 'Humidity'),
    'ph': (0, 14, 'pH'),
    'rainfall': (0, 300, 'Rainfall'),
}

# Load the trained model and label encoder
with open('crop_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_crop', methods=['POST'])
def predict_crop():
    try:
        # Get form data
        nitrogen = float(request.form['nitrogen'])
        phosphorus = float(request.form['phosphorus'])
        potassium = float(request.form['potassium'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # Validate input values
        errors = []
        input_values = {
            'nitrogen': nitrogen,
            'phosphorus': phosphorus,
            'potassium': potassium,
            'temperature': temperature,
            'humidity': humidity,
            'ph': ph,
            'rainfall': rainfall,
        }

        for field, value in input_values.items():
            min_value, max_value, label = INPUT_LIMITS[field]
            if value < min_value or value > max_value:
                errors.append(
                    f"Invalid {label} value. Please provide a value between {min_value} and {max_value}."
                )

        # If there are errors, return them
        if errors:
            return render_template('error.html', errors=errors), 400

        # Make prediction using the model
        features = pd.DataFrame([
            {
                'NITROGEN': nitrogen,
                'PHOSPHORUS': phosphorus,
                'POTASSIUM': potassium,
                'TEMPERATURE': temperature,
                'HUMIDITY': humidity,
                'PH': ph,
                'RAINFALL': rainfall,
            }
        ])
        predicted_crop = model.predict(features)

        # Convert the numerical prediction back to crop label
        crop_name = le.inverse_transform(predicted_crop)[0]

        return render_template('result.html', crop=crop_name)
    except ValueError:
        return render_template(
            'error.html',
            errors=["Invalid input type. Please provide numeric values for all fields."]
        ), 400
    except Exception as e:
        return render_template(
            'error.html',
            errors=[f"An error occurred: {str(e)}"]
        ), 500

if __name__ == '__main__':
    app.run(debug=True)
