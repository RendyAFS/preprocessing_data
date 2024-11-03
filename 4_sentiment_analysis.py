import nltk
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize

# Unduh lexicon VADER
nltk.downloader.download('vader_lexicon')

# Inisialisasi SentimentIntensityAnalyzer
senti_indo = SentimentIntensityAnalyzer()

# Memuat kamus lexicon berbahasa Indonesia
url = 'https://drive.google.com/file/d/1qPX0Uej3PqUQUI3op_oeEr8AdmrgOT2V/view?usp=sharing'
path = 'https://drive.google.com/uc?export=download&id=' + url.split('/')[-2]
df_senti = pd.read_csv(path, sep=':', names=['word', 'value'])

# Update lexicon VADER dengan kata-kata dari bahasa Indonesia
senti_dict = {row['word']: row['value'] for _, row in df_senti.iterrows()}
senti_indo.lexicon.update(senti_dict)

# Menambahkan kata tambahan
kata_tambahan = {
    "pudar": -5,
    "fast": 1,
}
senti_indo.lexicon.update(kata_tambahan)

print("Lexicon Bahasa Indonesia berhasil ditambahkan ke SentimentIntensityAnalyzer.")

# Membaca data preprocessed text dari CSV
input_path = 'Output/2_preprocessed_text_output.csv'
df_reviews = pd.read_csv(input_path)

# Mengisi NaN dengan string kosong dan konversi semua nilai menjadi string
df_reviews['preprocessed_text'] = df_reviews['preprocessed_text'].fillna('').astype(str)

# Menghitung skor sentimen untuk setiap teks
df_reviews['sentiment_score'] = df_reviews['preprocessed_text'].apply(
    lambda x: senti_indo.polarity_scores(x)['compound']
)

# Menyimpan hasil ke file CSV baru
output_path = 'Output/4_sentiment_analysis_output.csv'
df_reviews[['preprocessed_text', 'sentiment_score']].to_csv(output_path, index=False)

print(f"Analisis sentimen selesai, hasil disimpan ke '{output_path}'")