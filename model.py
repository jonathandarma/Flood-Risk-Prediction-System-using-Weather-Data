import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier # Dipakai sebagai proxy LSTM
import joblib

# --- 1. MEMBUAT DATASET DUMMY ---
# Data Cuaca: Curah Hujan (mm), Kelembapan (%), Ketinggian (mdpl)
# Target: Risiko Banjir (0=Low, 1=Medium, 2=High)
data_rf = {
    'Rainfall': [10, 25, 40, 60, 80, 110, 130, 150, 5, 30, 75, 140],
    'Humidity': [65, 70, 75, 80, 85, 90, 95, 98, 60, 70, 80, 90],
    'Elevation': [150, 120, 80, 60, 40, 20, 10, 5, 200, 150, 70, 30],
    'Risk': [0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 1, 2]
}
df_rf = pd.DataFrame(data_rf)

# --- 2. PELATIHAN MODEL RANDOM FOREST ---
X_rf = df_rf[['Rainfall', 'Humidity', 'Elevation']]
y_rf = df_rf['Risk']
model_rf = RandomForestClassifier(n_estimators=10, random_state=42)
model_rf.fit(X_rf, y_rf)
joblib.dump(model_rf, 'random_forest_model.pkl')
print("Model Random Forest berhasil dilatih.")

# --- 3. PELATIHAN MODEL LSTM DUMMY (MLP) ---
# LSTM membutuhkan data sekuensial. Kita simulasikan dengan data 3 hari (9 fitur)
# Misalnya: Day1_Rain, Day1_Hum, Day1_Elev, Day2_Rain, Day2_Hum, ...
data_lstm = {
    'd1_R': [10, 40, 70, 100, 10], 'd1_H': [60, 70, 80, 90, 60], 'd1_E': [100, 100, 100, 50, 50],
    'd2_R': [20, 50, 80, 110, 20], 'd2_H': [65, 75, 85, 95, 65], 'd2_E': [100, 100, 50, 50, 50],
    'd3_R': [30, 60, 90, 120, 30], 'd3_H': [70, 80, 90, 98, 70], 'd3_E': [100, 50, 50, 10, 10],
    'Risk': [0, 1, 1, 2, 0] # 0=Low, 1=Medium, 2=High
}
df_lstm = pd.DataFrame(data_lstm)

X_lstm = df_lstm.drop('Risk', axis=1)
y_lstm = df_lstm['Risk']

# MLPClassifier sebagai proxy untuk LSTM
model_lstm = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10,), random_state=42, max_iter=1000)
model_lstm.fit(X_lstm, y_lstm)
joblib.dump(model_lstm, 'lstm_model.pkl')
print("Model LSTM Dummy (MLP) berhasil dilatih dan disimpan sebagai 'lstm_model.pkl'")
