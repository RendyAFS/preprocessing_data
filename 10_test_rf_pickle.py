import joblib
import numpy as np

# Memuat model Random Forest dan vectorizer yang telah disimpan
model_filename = 'Output/random_forest_model.pkl'
vectorizer_filename = 'Output/tfidf_vectorizer.pkl'

# Muat model dan vectorizer
best_rf = joblib.load(model_filename)
vectorizer = joblib.load(vectorizer_filename)

# Fungsi untuk memprediksi sentimen
def predict_sentiment(input_text):
    # Transformasi input teks ke dalam vektor TF-IDF menggunakan vectorizer yang telah dilatih
    input_vector = vectorizer.transform([input_text])

    # Prediksi menggunakan model Random Forest
    prediction = best_rf.predict(input_vector)

    # Menyimpan label untuk hasil prediksi
    label_dict = {0: "Negatif", 1: "Netral", 2: "Positif"}
    return label_dict.get(prediction[0], "Unknown")

# Program utama untuk mengambil input dan memberikan prediksi
if __name__ == "__main__":
    # Ambil input teks dari pengguna
    input_text = input("Masukkan teks untuk prediksi sentimen: ")

    # Prediksi sentimen
    sentiment = predict_sentiment(input_text)

    # Tampilkan hasil prediksi
    print(f"Hasil prediksi sentimen: {sentiment}")

