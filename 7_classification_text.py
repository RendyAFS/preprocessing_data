import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Pastikan resource ini sudah diunduh
nltk.download('stopwords')
nltk.download('punkt')

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
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Bagi data menjadi data latih dan uji
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.25, random_state=42)

# # Model SVM dengan kernel linear
# model = SVC(kernel='linear', C=1.0)
# model.fit(X_train, y_train)

# Prediksi dan evaluasi
# y_pred = model.predict(X_test)
# print("Akurasi:", accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred, labels=[0, 1, 2], target_names=["Negatif", "Netral", "Positif"], zero_division=0))

# 1. SVM (Support Vector Machine)
print("Support Vector Machine (SVM)")
svm_clf = SVC(kernel='linear', C=1.0)
svm_clf.fit(X_train, y_train)
y_pred_svm = svm_clf.predict(X_test)
print("Akurasi:", accuracy_score(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm, labels=[0, 1, 2], target_names=["Negatif", "Netral", "Positif"], zero_division=0))

# 2. Logistic Regression
print("\nLogistic Regression")
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)
print("Akurasi:", accuracy_score(y_test, y_pred_log_reg))
print(classification_report(y_test, y_pred_log_reg, labels=[0, 1, 2], target_names=["Negatif", "Netral", "Positif"], zero_division=0))

# 3. Random Forest Classifier
print("\nRandom Forest Classifier")
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)
print("Akurasi:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf, labels=[0, 1, 2], target_names=["Negatif", "Netral", "Positif"], zero_division=0))

# 4. Naive Bayes Classifier (MultinomialNB)
print("\nNaive Bayes Classifier")
nb_clf = MultinomialNB()
nb_clf.fit(X_train, y_train)
y_pred_nb = nb_clf.predict(X_test)
print("Akurasi:", accuracy_score(y_test, y_pred_nb))
print(classification_report(y_test, y_pred_nb, labels=[0, 1, 2], target_names=["Negatif", "Netral", "Positif"], zero_division=0))

# 5. K-Nearest Neighbors (KNN)
print("\nK-Nearest Neighbors (KNN)")
knn_clf = KNeighborsClassifier(n_neighbors=5)
knn_clf.fit(X_train, y_train)
y_pred_knn = knn_clf.predict(X_test)
print("Akurasi:", accuracy_score(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn, labels=[0, 1, 2], target_names=["Negatif", "Netral", "Positif"], zero_division=0))


df_test = data.iloc[y_test.index].copy()  # Menggunakan index yang sama dengan data uji

# Menambah kolom untuk setiap hasil prediksi model
df_test['predicted_label_svm'] = y_pred_svm
df_test['predicted_label_log_reg'] = y_pred_log_reg
df_test['predicted_label_rf'] = y_pred_rf
df_test['predicted_label_nb'] = y_pred_nb
df_test['predicted_label_knn'] = y_pred_knn

# Menampilkan DataFrame dengan kolom baru
print(df_test.head())

# Menyimpan DataFrame ke file CSV jika diperlukan
output_file_path = 'Output/7_classification_text.csv'  # Tentukan path untuk file output
df_test.to_csv(output_file_path, index=False)
print(f"Hasil prediksi telah disimpan ke {output_file_path}")