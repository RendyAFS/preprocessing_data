import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib  # Import joblib untuk menyimpan model

# Load dataset
file_path = 'Output/6_lexicon_labeled_number.csv'
data = pd.read_csv(file_path)

# Check if required columns exist
if 'preprocessed_text' not in data.columns or 'label_sentiment_number' not in data.columns:
    raise ValueError("Kolom 'preprocessed_text' atau 'label_sentiment_number' tidak ditemukan dalam CSV!")

texts = data['preprocessed_text'].astype(str)
labels = data['label_sentiment_number'].astype(int)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(texts)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.25, random_state=42)

# Random Forest with GridSearchCV
print("Random Forest Classifier")
rf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
rf_grid = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    rf_params,
    n_iter=10,
    cv=3,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
)
rf_grid.fit(X_train, y_train)
best_rf = rf_grid.best_estimator_

# Predictions
y_pred_rf = best_rf.predict(X_test)

# Evaluation
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print(f"Best Parameters for Random Forest: {rf_grid.best_params_}")
print("Akurasi:", rf_accuracy)
print(classification_report(y_test, y_pred_rf, labels=[0, 1, 2], target_names=["Negatif", "Netral", "Positif"], zero_division=0))

# Save evaluation results
evaluation_results = pd.DataFrame([{
    'Model': 'Random Forest Classifier',
    'Accuracy': rf_accuracy,
    'Best Parameters': rf_grid.best_params_
}])

output_eval_file = 'Output/8_evaluation_accuracy_random_forest.csv'
evaluation_results.to_csv(output_eval_file, index=False)

# Print evaluation results
print("\nHasil Evaluasi Model:")
print(evaluation_results)
print(f"Hasil evaluasi akurasi telah disimpan ke {output_eval_file}")

# Save the trained model and vectorizer to .pkl files
model_filename = 'Output/random_forest_model.pkl'
vectorizer_filename = 'Output/tfidf_vectorizer.pkl'

joblib.dump(best_rf, model_filename)  # Save Random Forest model
joblib.dump(vectorizer, vectorizer_filename)  # Save TF-IDF vectorizer

# Log model and vectorizer saving
print(f"Model telah disimpan ke {model_filename}")
print(f"Vectorizer telah disimpan ke {vectorizer_filename}")
