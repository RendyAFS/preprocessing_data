import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


# Baca file CSV
file_path = 'Output/6_lexicon_labeled_number.csv'  # Ganti dengan path ke file CSV Anda
data = pd.read_csv(file_path)

# Pastikan kolom ada di CSV
if 'preprocessed_text' not in data.columns or 'label_sentiment_number' not in data.columns:
    raise ValueError("Kolom 'preprocessed_text' atau 'label_sentiment_number' tidak ditemukan dalam CSV!")

# Ambil teks dan label
texts = data['preprocessed_text'].astype(str)  # Pastikan kolom ini berupa string
labels = data['label_sentiment_number'].astype(int)  # Pastikan label berupa angka

# Vektorisasi teks menggunakan TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(texts)

# Bagi data menjadi data latih dan uji
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.25, random_state=42)

# Random Forest Classifier
print("\nRandom Forest Classifier")
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)

# Evaluasi model
print("Akurasi:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf, labels=[0, 1, 2], target_names=["Negatif", "Netral", "Positif"], zero_division=0))

# Plotting Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, model_name):
    """Plot confusion matrix for a given model."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Negatif", "Netral", "Positif"], yticklabels=["Negatif", "Netral", "Positif"])
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

plot_confusion_matrix(y_test, y_pred_rf, "Random Forest")

# Plotting Learning Curve
def plot_learning_curve_with_loss(model, X_train, y_train, model_name):
    """Plot learning curve and proxy loss for a given model."""
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train, y_train, cv=5, train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    # Compute loss as 1 - accuracy (proxy for training and validation loss)
    train_loss = 1 - train_scores_mean
    test_loss = 1 - test_scores_mean

    # Plot Training and Validation Accuracy
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_sizes, train_scores_mean, label="Training Accuracy", color="blue")
    plt.plot(train_sizes, test_scores_mean, label="Validation Accuracy", color="green")
    plt.title(f"Training and Validation Accuracy - {model_name}")
    plt.xlabel("Training Size")
    plt.ylabel("Accuracy")
    plt.legend(loc="best")

    # Plot Training and Validation Loss
    plt.subplot(1, 2, 2)
    plt.plot(train_sizes, train_loss, label="Training Loss", color="blue")
    plt.plot(train_sizes, test_loss, label="Validation Loss", color="green")
    plt.title(f"Training and Validation Loss - {model_name}")
    plt.xlabel("Training Size")
    plt.ylabel("Loss")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()

plot_learning_curve_with_loss(rf_clf, X_train, y_train, "Random Forest")
