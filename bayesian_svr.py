import scipy.io
from sklearn.model_selection import train_test_split, KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

# Veri Setlerini Yükleme
c3_data = scipy.io.loadmat('C:/Users/burak/OneDrive/Masaüstü/ÖRÜNTÜ2024_Proje/C3/wifi_localization.mat')
feat_c3 = c3_data['feat']
lbl_c3 = c3_data['lbl']

# Veri Setlerini Kontrol Etme
print("\nC3 Veri Seti:")
print("feat_c3 shape:", feat_c3.shape)
print("lbl_c3 shape:", lbl_c3.shape)

# Farklı test oranlarını belirleyin
test_sizes = [0.2, 0.3, 0.4]

for test_size in test_sizes:
    print(f"\nTest Size: {test_size}")

    # Veriyi eğitim ve test alt kümelerine ayırın
    feat_c3_train, feat_c3_test, lbl_c3_train, lbl_c3_test = train_test_split(feat_c3, lbl_c3.ravel(), test_size=test_size, random_state=42)

    # Bayesian Sınıflandırma Modeli
    bayesian_model = GaussianNB()
    bayesian_model.fit(feat_c3_train, lbl_c3_train)

    # Test verisi üzerinde tahmin yap
    bayesian_pred = bayesian_model.predict(feat_c3_test)

    # ACC ve F-Score hesapla
    acc_test = accuracy_score(lbl_c3_test, bayesian_pred)
    f1_test = f1_score(lbl_c3_test, bayesian_pred, average='weighted')

    # Sonuçları yazdırın
    print("\nBayesian Classification Scores:")
    print("Accuracy:", acc_test)
    print("F1 Score:", f1_test)
