import pandas as pd
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import pickle

# Load dataset and remove duplicate rows to avoid train/test leakage.
data = pd.read_csv('crop.csv').drop_duplicates().reset_index(drop=True)

# Features and target
X = data[['NITROGEN', 'PHOSPHORUS', 'POTASSIUM', 'TEMPERATURE', 'HUMIDITY', 'PH', 'RAINFALL']]
y = data['CROP']

# Encode the target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split the dataset for final holdout reporting and model persistence.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.3,
    random_state=42,
    stratify=y_encoded
)

# Models to compare
ensemble_model = VotingClassifier(
    estimators=[
        ('knn', Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(n_neighbors=11))
        ])),
        ('rf', RandomForestClassifier(
            n_estimators=120,
            max_depth=12,
            min_samples_leaf=2,
            random_state=42
        )),
        ('logreg', Pipeline([
            ('scaler', StandardScaler()),
            ('logreg', LogisticRegression(C=0.5, max_iter=1000, random_state=42))
        ]))
    ],
    voting='soft'
)

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=120,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42
    ),
    "Logistic Regression": Pipeline([
        ('scaler', StandardScaler()),
        ('logreg', LogisticRegression(C=0.5, max_iter=1000, random_state=42))
    ]),
    "KNN": Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=11))
    ]),
    "Voting Ensemble (KNN + RF + LR)": ensemble_model
}

# Repeated stratified CV gives a more realistic estimate than one split.
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
scoring = {
    'accuracy': 'accuracy',
    'precision_macro': 'precision_macro',
    'recall_macro': 'recall_macro',
    'f1_macro': 'f1_macro'
}

cv_results_summary = {}


def print_metrics_table(title, y_true, y_pred):
    """Print ensemble metrics with metrics on the rows and class columns on the x axis."""
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    class_labels = [label for label in report.keys() if label not in {'accuracy', 'macro avg', 'weighted avg'}]
    display_columns = ['Range', 'Squeeze', 'Breakout', 'Macro Avg']

    table = pd.DataFrame(index=['Accuracy', 'Precision', 'Recall', 'F1 Score'], columns=display_columns)

    for display_name, class_label in zip(display_columns[:-1], class_labels[:3]):
        class_metrics = report[class_label]
        table.loc['Accuracy', display_name] = '—'
        table.loc['Precision', display_name] = class_metrics['precision']
        table.loc['Recall', display_name] = class_metrics['recall']
        table.loc['F1 Score', display_name] = class_metrics['f1-score']

    macro_avg = report['macro avg']
    table.loc['Accuracy', 'Macro Avg'] = report['accuracy']
    table.loc['Precision', 'Macro Avg'] = macro_avg['precision']
    table.loc['Recall', 'Macro Avg'] = macro_avg['recall']
    table.loc['F1 Score', 'Macro Avg'] = macro_avg['f1-score']

    print(f"\n{title}")
    print(table.to_string(float_format=lambda value: f"{value:.4f}" if isinstance(value, (int, float)) else str(value)))

# Evaluate each model with cross-validation first.
for name, model in models.items():
    cv_scores = cross_validate(model, X, y_encoded, cv=cv, scoring=scoring, n_jobs=-1)
    cv_results_summary[name] = {
        'accuracy': cv_scores['test_accuracy'].mean(),
        'precision': cv_scores['test_precision_macro'].mean(),
        'recall': cv_scores['test_recall_macro'].mean(),
        'f1': cv_scores['test_f1_macro'].mean()
    }

    print(f"\n{name} CV Results (Mean):")
    print(f"Accuracy : {cv_results_summary[name]['accuracy']:.4f}")
    print(f"Precision: {cv_results_summary[name]['precision']:.4f}")
    print(f"Recall   : {cv_results_summary[name]['recall']:.4f}")
    print(f"F1 Score : {cv_results_summary[name]['f1']:.4f}")

# Also evaluate on holdout test split for a single-run snapshot.
holdout_results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    holdout_results[name] = {
        'accuracy': accuracy_score(y_test, predictions),
        'precision': precision_score(y_test, predictions, average='macro', zero_division=0),
        'recall': recall_score(y_test, predictions, average='macro', zero_division=0),
        'f1': f1_score(y_test, predictions, average='macro', zero_division=0)
    }

    if name == "Voting Ensemble (KNN + RF + LR)":
        print_metrics_table(
            "Voting Ensemble Holdout Metrics",
            y_test,
            predictions,
        )
    else:
        print(f"\n{name} Holdout Results:")
        print(f"Accuracy : {holdout_results[name]['accuracy']:.4f}")
        print(f"Precision: {holdout_results[name]['precision']:.4f}")
        print(f"Recall   : {holdout_results[name]['recall']:.4f}")
        print(f"F1 Score : {holdout_results[name]['f1']:.4f}")

# Save the best model using CV accuracy (more robust than one split).
best_model_name = max(cv_results_summary, key=lambda model_name: cv_results_summary[model_name]['accuracy'])
best_model = models[best_model_name]
best_model.fit(X_train, y_train)

with open('best_crop_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print("\nComparison complete. Best model:", best_model_name)

# Save ensemble model to app's expected filename as requested.
ensemble_model.fit(X_train, y_train)
with open('crop_model.pkl', 'wb') as f:
    pickle.dump(ensemble_model, f)

print("Ensemble model saved to crop_model.pkl")
