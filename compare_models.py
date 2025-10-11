import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
data = pd.read_csv('crop.csv')

# Features and target
X = data[['NITROGEN', 'PHOSPHORUS', 'POTASSIUM', 'TEMPERATURE', 'HUMIDITY', 'PH', 'RAINFALL']]
y = data['CROP']

# Encode the target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Models to compare
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=200),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

# Dictionary to store accuracy results
accuracy_results = {}

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    accuracy_results[name] = acc
    print(f"{name} Accuracy: {acc:.4f}")

# Save the best model
best_model_name = max(accuracy_results, key=accuracy_results.get)
best_model = models[best_model_name]

with open('best_crop_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print("\nComparison complete. Best model:", best_model_name)
