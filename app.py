from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

INPUT_LIMITS = {
    'nitrogen': (0, 140, 'nitrogen'),
    'phosphorus': (0, 145, 'phosphorus'),
    'potassium': (0, 205, 'potassium'),
    'temperature': (0, 50, 'temperature'),
    'humidity': (0, 100, 'humidity'),
    'ph': (0, 14, 'pH'),
    'rainfall': (0, 300, 'rainfall'),
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
        # Parse and validate all fields first so every bad input lands on error page.
        input_values = {}
        errors = []
        invalid_type_detected = False

        for field, (_, _, label) in INPUT_LIMITS.items():
            raw_value = request.form.get(field, '').strip()
            if raw_value == '':
                errors.append(f"{label} is required.")
                continue

            try:
                value = float(raw_value)
            except ValueError:
                invalid_type_detected = True
                errors.append(f"Invalid {label} value. Please provide a numeric value.")
                continue

            min_value, max_value, _ = INPUT_LIMITS[field]
            if value < min_value or value > max_value:
                errors.append(
                    f"Invalid {label} value. Please provide a value between {min_value} and {max_value}."
                )
            else:
                input_values[field] = value

        if invalid_type_detected:
            errors.insert(0, "Invalid input type. Please provide numeric values for all fields.")

        if errors:
            return render_template('error.html', errors=errors), 400

        # Make prediction using the model
        features = pd.DataFrame([
            {
                'NITROGEN': input_values['nitrogen'],
                'PHOSPHORUS': input_values['phosphorus'],
                'POTASSIUM': input_values['potassium'],
                'TEMPERATURE': input_values['temperature'],
                'HUMIDITY': input_values['humidity'],
                'PH': input_values['ph'],
                'RAINFALL': input_values['rainfall'],
            }
        ])
        predicted_crop = model.predict(features)

        # Convert the numerical prediction back to crop label
        crop_name = le.inverse_transform(predicted_crop)[0]

        return render_template('result.html', crop=crop_name)
    except Exception as e:
        return render_template(
            'error.html',
            errors=[f"An error occurred: {str(e)}"]
        ), 500

if __name__ == '__main__':
    app.run(debug=True)
