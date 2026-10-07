import matplotlib
matplotlib.use('Agg')  # must be set before anything imports matplotlib.pyplot

from flask import Flask, render_template, request
import pandas as pd
import pickle
import os
import uuid
from werkzeug.utils import secure_filename

import satellite
import gee_utils

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('static', exist_ok=True)

INPUT_LIMITS = {
    'nitrogen': (0, 140, 'nitrogen'),
    'phosphorus': (0, 145, 'phosphorus'),
    'potassium': (0, 205, 'potassium'),
    'temperature': (0, 50, 'temperature'),
    'humidity': (0, 100, 'humidity'),
    'ph': (0, 14, 'pH'),
    'rainfall': (0, 300, 'rainfall'),
}


def get_ndvi_assessment(ndvi_mean):
    """Return a generic vegetation-condition assessment from mean NDVI."""
    score = round(max(0, min(100, ((ndvi_mean + 1) / 2) * 100)))

    if ndvi_mean < 0.2:
        label = 'Low'
        explanation = (
            'Existing vegetation activity is low. The area may contain bare soil, '
            'water, sparse vegetation, or stressed plants.'
        )
    elif ndvi_mean < 0.5:
        label = 'Moderate'
        explanation = 'Existing vegetation activity is moderate and may indicate sparse or developing vegetation.'
    elif ndvi_mean < 0.7:
        label = 'Good'
        explanation = 'Existing vegetation activity is good and indicates established vegetation.'
    else:
        label = 'High'
        explanation = 'Existing vegetation activity is high and indicates dense, healthy vegetation.'

    return {
        'score': score,
        'label': label,
        'explanation': explanation,
    }

# Load the trained model and label encoder
with open('crop_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# Attempt to initialize Google Earth Engine for the Flask process so request handlers
# can fetch tiles. Use the static project as requested.
gee_project = 'crop-recommendation-506706'
try:
    gee_utils.initialize_ee(gee_project)
    print(f'Google Earth Engine initialized for project: {gee_project}')
except Exception as e:
    # Do not crash the app; surface a clear message when fetch is attempted.
    print('Warning: Earth Engine initialization failed at startup:', str(e))


@app.route('/ee_status')
def ee_status():
    """Return basic Earth Engine initialization info: project and service-account (if any)."""
    sa = gee_utils.get_service_account_email()
    return {
        'project': gee_project,
        'service_account': sa,
        'note': 'Ensure the active identity (service account or interactive user) has permission on this project.'
    }

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

        # Handle optional satellite image upload (GeoTIFF with Red+NIR bands)
        ndvi_info = None
        sat_file = request.files.get('sat_image')
        if sat_file and sat_file.filename != '':
            filename = secure_filename(sat_file.filename)
            save_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}_{filename}")
            sat_file.save(save_path)
            try:
                ndvi_info = satellite.compute_ndvi_from_geotiff(save_path, out_dir='static')
            except Exception as e:
                # surface the satellite processing error to the user
                return render_template('error.html', errors=[f"Satellite image processing error: {str(e)}"]), 400
            finally:
                try:
                    os.remove(save_path)
                except Exception:
                    pass

        # If no uploaded file, but latitude/longitude provided, try to fetch from Earth Engine
        if ndvi_info is None:
            lat = request.form.get('latitude', '').strip()
            lon = request.form.get('longitude', '').strip()
            if lat and lon:
                try:
                    fetched = gee_utils.fetch_sentinel2_red_nir(lat, lon, out_dir='static')
                    tif_path = fetched.get('path')
                    ndvi_info = satellite.compute_ndvi_from_geotiff(tif_path, out_dir='static')
                    # remove the downloaded tif after processing
                    try:
                        os.remove(tif_path)
                    except Exception:
                        pass
                except Exception as e:
                    return render_template('error.html', errors=[f"Earth Engine fetch error: {str(e)}"]), 400

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
        ndvi_assessment = get_ndvi_assessment(ndvi_info['mean']) if ndvi_info else None

        return render_template(
            'result.html',
            crop=crop_name,
            ndvi_mean=(ndvi_info['mean'] if ndvi_info else None),
            ndvi_std=(ndvi_info['std'] if ndvi_info else None),
            ndvi_image=(ndvi_info['image'] if ndvi_info else None),
            ndvi_assessment=ndvi_assessment,
        )
    except Exception as e:
        return render_template(
            'error.html',
            errors=[f"An error occurred: {str(e)}"]
        ), 500

if __name__ == '__main__':
    app.run(debug=True)