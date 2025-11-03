from flask import Flask, request, jsonify
from flask_cors import CORS # Penting untuk koneksi antara web dan server
import joblib
import numpy as np

app = Flask(__name__)
# Mengizinkan akses dari semua sumber (untuk deployment lokal/demo)
CORS(app) 

# Memuat model Random Forest yang sudah dilatih
try:
    model = joblib.load('random_forest_model.pkl')
    RISK_LABELS = {0: 'LOW', 1: 'MEDIUM', 2: 'HIGH'}
    print("Model AI berhasil dimuat.")
except FileNotFoundError:
    print("ERROR: File 'random_forest_model.pkl' tidak ditemukan. Jalankan model.py terlebih dahulu.")
    model = None

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model AI tidak tersedia.'}), 500

    # Mendapatkan data JSON yang dikirim dari JavaScript
    data = request.get_json(force=True)

    try:
        # Mengambil fitur yang dibutuhkan model
        rainfall = data['rainfall']
        humidity = data['humidity']
        elevation = data['elevation']
        
        # Membuat array input yang sesuai dengan format training model
        features = np.array([[rainfall, humidity, elevation]])

        # Melakukan Prediksi menggunakan Model Random Forest
        prediction_int = model.predict(features)[0]
        prediction_prob = model.predict_proba(features) # Probabilitas

        # Mengubah hasil integer menjadi label
        risk_level = RISK_LABELS.get(prediction_int, 'UNKNOWN')
        
        # Mengambil probabilitas risiko tertinggi
        highest_prob = np.max(prediction_prob) * 100

        # Mengembalikan hasil prediksi ke JavaScript (index.html)
        return jsonify({
            'risk_level': risk_level,
            'probability': round(highest_prob, 2),
            'status': 'success'
        })
    except Exception as e:
        # Menangani kesalahan input
        return jsonify({'error': f'Terjadi kesalahan saat prediksi: {str(e)}'}), 400

if __name__ == '__main__':
    # Jalankan server Flask pada port 5000
    print("Server Flask siap menerima permintaan di http://127.0.0.1:5000/predict")
    app.run(port=5000, debug=True)
