import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
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

# Build a soft-voting ensemble with three different model families.
knn_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=7))
])

logreg_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(max_iter=1000, random_state=42))
])

rf_model = RandomForestClassifier(n_estimators=200, random_state=42)

model = VotingClassifier(
    estimators=[
        ('knn', knn_pipeline),
        ('rf', rf_model),
        ('logreg', logreg_pipeline)
    ],
    voting='soft'
)

# Train ensemble model
model.fit(X_train, y_train)

# Evaluate quickly for training visibility
test_accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Ensemble test accuracy: {test_accuracy:.4f}")

# Save the trained model and label encoder for later use in the web app
with open('crop_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print("Model training complete and saved!")
