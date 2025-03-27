import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# 1. Veri Setini Yükleme
data = pd.read_csv('heart.csv')
data.head()

# 2. Genel Bakış
print(data.head())
print(data.info())

# 3. Eksik Veri Kontrolü
print("\nEksik veri kontrolü:")
print(data.isnull().sum())

# 4. Eksik veri varsa, eksik verileri doldurma
data.fillna(data.mean(), inplace=True)


# 5. Temel İstatistikleri Görüntüleme
print("\nTemel İstatistikler:")
print(data.describe())

# 6. Veriyi Eğitim ve Test Olarak Ayırma ve Ölçeklendirme
def split_and_scale_data(data):
    x = data.drop("target", axis=1)  # "target" sütununu ayır
    y = data["target"]
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    # StandardScaler ile ölçeklendirme
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    
    # NumPy dizisini tekrar DataFrame'e çevir ve sütun isimlerini koru
    x_train_scaled = pd.DataFrame(x_train_scaled, columns=x.columns)
    x_test_scaled = pd.DataFrame(x_test_scaled, columns=x.columns)

    return x_train_scaled, x_test_scaled, y_train, y_test, scaler

# Fonksiyonu çağır
x_train, x_test, y_train, y_test, scaler = split_and_scale_data(data)

# Eğitim verisini düzgün bastırmak için DataFrame'e dönüştürelim
train_data = x_train.copy()
train_data["target"] = y_train.reset_index(drop=True)  # Hedef sütunu tekrar ekleyelim

print("\nKalp krizi testi (Eğitim Verisi İlk 5 Satır):")
print(train_data.head())  # x_train ve y_train birlikte gösterilecek

print(f"\nEğitim: {x_train.shape}, Test: {x_test.shape}")  # Boyutları göster

# 7. Modeli Eğitme
model = LogisticRegression()
model.fit(x_train, y_train)

new_patient = pd.DataFrame({
    "age": [int(input("Yaş: "))],
    "sex": [int(input("Cinsiyet (0: Kadın, 1: Erkek): "))],
    "cp": [int(input("Göğüs ağrısı tipi (0-3): "))],
    "trestbps": [int(input("Dinlenme anındaki kan basıncı (94 - 200mm Hg): "))],
    "chol": [int(input("Kolesterol(126 - 564mgdl): "))],
    "fbs": [int(input("Açlık kan şekeri > 120 mg/dl (1: Evet, 0: Hayır): "))],
    "restecg": [int(input("Elektrokardiyografik sonuçlar (0-2): "))],
    "thalach": [int(input("Maksimum kalp atış hızı(71 - 202): "))],
    "exang": [int(input("Egzersize bağlı anjin (1: Evet, 0: Hayır): "))],
    "oldpeak": [float(input("Egzersiz ile indüklenen ST depresyonu(0.0 - 6.2): "))],
    "slope": [int(input("Eğim(0 - 2): "))],
    "ca": [int(input("Kan sayımı: "))],
    "thal": [int(input("Thalassemi: "))]
})

new_patient_scaled = scaler.transform(new_patient)

prediction = model.predict(new_patient)
if prediction[0] == 1:
    print("Hasta kalp krizi riski altında!")
else:
    print("Hasta kalp krizi riski altında değil!")
