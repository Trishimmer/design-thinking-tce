import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
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
ensemble_model = VotingClassifier(
    estimators=[
        ('knn', Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(n_neighbors=7))
        ])),
        ('rf', RandomForestClassifier(n_estimators=200, random_state=42)),
        ('logreg', Pipeline([
            ('scaler', StandardScaler()),
            ('logreg', LogisticRegression(max_iter=1000, random_state=42))
        ]))
    ],
    voting='soft'
)

models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=200),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Voting Ensemble (KNN + RF + LR)": ensemble_model
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

# Save ensemble model to app's expected filename as requested.
ensemble_model.fit(X_train, y_train)
with open('crop_model.pkl', 'wb') as f:
    pickle.dump(ensemble_model, f)
