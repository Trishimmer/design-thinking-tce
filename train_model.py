import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# Load dataset
data = pd.read_csv('crop.csv')

# Preprocess the data
X = data[['NITROGEN', 'PHOSPHORUS', 'POTASSIUM', 'TEMPERATURE', 'HUMIDITY', 'PH', 'RAINFALL']]
y = data['CROP']

# Convert crop labels to numbers
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model and label encoder for later use in the web app
with open('crop_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print("Model training complete and saved!")
