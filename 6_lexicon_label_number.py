import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import re

# Unduh lexicon VADER
nltk.download('vader_lexicon')

# Path ke file lexicon bahasa Indonesia
path = 'Data/sentiwords_id.txt'  # Pastikan file ini sudah ada
df_senti = pd.read_csv(path, sep=':', names=['word', 'value'])

# Buat dictionary dari lexicon bahasa Indonesia
senti_dict = {row['word']: float(row['value']) for _, row in df_senti.iterrows()}

# Inisialisasi Stemmer dari Sastrawi
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Inisialisasi SentimentIntensityAnalyzer dan update lexicon-nya
senti_indo = SentimentIntensityAnalyzer()
senti_indo.lexicon.update(senti_dict)

# Load data dari 'preprocessed_text_output.csv'
df = pd.read_csv('Output/2_preprocessed_text_output.csv')  # Pastikan file ini sudah ada

# Pastikan kolom 'preprocessed_text' adalah string
df['preprocessed_text'] = df['preprocessed_text'].astype(str)

# List untuk menyimpan hasil sentimen
label_lexicon = []

# Fungsi untuk tokenisasi menggunakan regex
def tokenize(text):
    return re.findall(r'\w+', text)

# Iterasi melalui setiap baris di DataFrame
for index, row in df.iterrows():
    # Stemming pada teks
    stemmed_text = stemmer.stem(row['preprocessed_text'])
    
    # Hitung skor sentimen untuk teks yang sudah distem
    score = senti_indo.polarity_scores(stemmed_text)
    
    # Tentukan label sentimen dengan kata
    if score['compound'] >= 0.05:
        label_lexicon.append(2)  # positif
    elif score['compound'] <= -0.05:
        label_lexicon.append(0)  # negatif
    else:
        label_lexicon.append(1)  # netral

# Tambahkan hasil sentimen sebagai kolom baru di DataFrame
df['label_sentiment_number'] = label_lexicon

# Simpan DataFrame yang sudah diberi label ke file CSV baru
df.to_csv('Output/6_lexicon_labeled_number.csv', index=False)

print("Proses selesai! Hasil sentimen telah disimpan di '6_lexicon_labeled_number.csv'")