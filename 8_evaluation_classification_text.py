import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

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

# Prepare a dictionary to store evaluation results
evaluation_results = []

# 1. SVM with GridSearchCV
print("Support Vector Machine (SVM)")
svm_params = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}
svm_grid = RandomizedSearchCV(SVC(), svm_params, n_iter=10, cv=3, scoring='accuracy', random_state=42, n_jobs=-1)
svm_grid.fit(X_train, y_train)
best_svm = svm_grid.best_estimator_
y_pred_svm = best_svm.predict(X_test)
svm_accuracy = accuracy_score(y_test, y_pred_svm)
print(f"Best Parameters for SVM: {svm_grid.best_params_}")
print("Akurasi:", svm_accuracy)
print(classification_report(y_test, y_pred_svm, labels=[0, 1, 2], target_names=["Negatif", "Netral", "Positif"], zero_division=0))
evaluation_results.append({'Model': 'Support Vector Machine (SVM)', 'Accuracy': svm_accuracy})

# 2. Logistic Regression with GridSearchCV
print("\nLogistic Regression")
log_reg_params = {
    'C': [0.1, 1, 10],
    'solver': ['lbfgs', 'liblinear'],
    'class_weight': ['balanced', None]
}
log_reg_grid = RandomizedSearchCV(LogisticRegression(max_iter=1000), log_reg_params, n_iter=10, cv=3, scoring='accuracy', random_state=42, n_jobs=-1)
log_reg_grid.fit(X_train, y_train)
best_log_reg = log_reg_grid.best_estimator_
y_pred_log_reg = best_log_reg.predict(X_test)
log_reg_accuracy = accuracy_score(y_test, y_pred_log_reg)
print(f"Best Parameters for Logistic Regression: {log_reg_grid.best_params_}")
print("Akurasi:", log_reg_accuracy)
print(classification_report(y_test, y_pred_log_reg, labels=[0, 1, 2], target_names=["Negatif", "Netral", "Positif"], zero_division=0))
evaluation_results.append({'Model': 'Logistic Regression', 'Accuracy': log_reg_accuracy})

# 3. Random Forest with GridSearchCV
print("\nRandom Forest Classifier")
rf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
rf_grid = RandomizedSearchCV(RandomForestClassifier(random_state=42), rf_params, n_iter=10, cv=3, scoring='accuracy', random_state=42, n_jobs=-1)
rf_grid.fit(X_train, y_train)
best_rf = rf_grid.best_estimator_
y_pred_rf = best_rf.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print(f"Best Parameters for Random Forest: {rf_grid.best_params_}")
print("Akurasi:", rf_accuracy)
print(classification_report(y_test, y_pred_rf, labels=[0, 1, 2], target_names=["Negatif", "Netral", "Positif"], zero_division=0))
evaluation_results.append({'Model': 'Random Forest Classifier', 'Accuracy': rf_accuracy})

# 4. Naive Bayes
print("\nNaive Bayes Classifier")
nb_params = {'alpha': [0.1, 0.5, 1.0, 2.0]}
nb_grid = RandomizedSearchCV(MultinomialNB(), nb_params, n_iter=10, cv=3, scoring='accuracy', random_state=42, n_jobs=-1)
nb_grid.fit(X_train, y_train)
best_nb = nb_grid.best_estimator_
y_pred_nb = best_nb.predict(X_test)
nb_accuracy = accuracy_score(y_test, y_pred_nb)
print(f"Best Parameters for Naive Bayes: {nb_grid.best_params_}")
print("Akurasi:", nb_accuracy)
print(classification_report(y_test, y_pred_nb, labels=[0, 1, 2], target_names=["Negatif", "Netral", "Positif"], zero_division=0))
evaluation_results.append({'Model': 'Naive Bayes Classifier', 'Accuracy': nb_accuracy})

# 5. KNN with GridSearchCV
print("\nK-Nearest Neighbors (KNN)")
knn_params = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}
knn_grid = RandomizedSearchCV(KNeighborsClassifier(), knn_params, n_iter=10, cv=3, scoring='accuracy', random_state=42, n_jobs=-1)
knn_grid.fit(X_train, y_train)
best_knn = knn_grid.best_estimator_
y_pred_knn = best_knn.predict(X_test)
knn_accuracy = accuracy_score(y_test, y_pred_knn)
print(f"Best Parameters for KNN: {knn_grid.best_params_}")
print("Akurasi:", knn_accuracy)
print(classification_report(y_test, y_pred_knn, labels=[0, 1, 2], target_names=["Negatif", "Netral", "Positif"], zero_division=0))
evaluation_results.append({'Model': 'K-Nearest Neighbors (KNN)', 'Accuracy': knn_accuracy})

# Create a DataFrame for evaluation results
df_evaluation = pd.DataFrame(evaluation_results)

# Save results to CSV
output_eval_file = 'Output/8_evaluation_accuracy_tuned.csv'
df_evaluation.to_csv(output_eval_file, index=False)

# Print evaluation results
print("\nHasil Evaluasi Model setelah Tuning:")
print(df_evaluation)
print(f"Hasil evaluasi akurasi telah disimpan ke {output_eval_file}")
