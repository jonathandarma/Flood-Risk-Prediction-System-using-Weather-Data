import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

# --- 1. MEMBUAT DATASET DUMMY (SIMULASI DATA CUACA) ---
# Data Cuaca: Curah Hujan (mm), Kelembapan (%), Ketinggian (mdpl)
# Target: Risiko Banjir (0=Low, 1=Medium, 2=High)
data = {
    'Rainfall': [10, 25, 40, 60, 80, 110, 130, 150, 5, 30, 75, 140],
    'Humidity': [65, 70, 75, 80, 85, 90, 95, 98, 60, 70, 80, 90],
    'Elevation': [150, 120, 80, 60, 40, 20, 10, 5, 200, 150, 70, 30],
    'Risk': [0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 1, 2]
}
df = pd.DataFrame(data)

# --- 2. PELATIHAN MODEL RANDOM FOREST ---
X = df[['Rainfall', 'Humidity', 'Elevation']]
y = df['Risk']

# Inisialisasi dan latih model Random Forest
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X, y)

# --- 3. SIMPAN MODEL ---
# Menyimpan model yang sudah terlatih untuk digunakan di file app.py
joblib.dump(model, 'random_forest_model.pkl')
print("Model Random Forest berhasil dilatih dan disimpan sebagai 'random_forest_model.pkl'")
