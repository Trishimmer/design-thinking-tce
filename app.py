from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

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
        if nitrogen < 0 or nitrogen > 140:
            errors.append("Invalid nitrogen value. Please provide a value between 0 and 140.")
        if phosphorus <= 0:
            errors.append("Invalid phosphorus value. Please provide a positive value.")
        if potassium <= 0:
            errors.append("Invalid potassium value. Please provide a positive value.")
        if temperature < 0 or temperature > 37:
            errors.append("Invalid temperature value. Please provide a value between 0 and 37.")
        if humidity <= 0:
            errors.append("Invalid humidity value. Please provide a positive value.")
        if ph < 1.0 or ph > 10.0:
            errors.append("Invalid pH value. Please provide a value between 1 and 10.")
        if rainfall < 0 or rainfall > 199:
            errors.append("Invalid rainfall value. Please provide a value between 0 and 199.")

        # If there are errors, return them
        if errors:
            return jsonify({"errors": errors}), 400

        # Make prediction using the model
        features = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
        predicted_crop = model.predict(features)

        # Convert the numerical prediction back to crop label
        crop_name = le.inverse_transform(predicted_crop)[0]

        return render_template('result.html', crop=crop_name)
    except ValueError:
        return jsonify({"error": "Invalid input type. Please provide numeric values."}), 400
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
