from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app) 

# Memuat kedua model yang sudah dilatih
try:
    model_rf = joblib.load('random_forest_model.pkl')
    model_lstm = joblib.load('lstm_model.pkl')
    RISK_LABELS = {0: 'LOW', 1: 'MEDIUM', 2: 'HIGH'}
    print("Kedua Model AI (RF dan LSTM) berhasil dimuat.")
except FileNotFoundError:
    print("ERROR: Pastikan 'random_forest_model.pkl' dan 'lstm_model.pkl' ada. Jalankan model.py terlebih dahulu.")
    model_rf = None
    model_lstm = None

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    model_choice = data.get('model_choice') # Ambil pilihan model dari frontend

    if model_choice == 'RF':
        model_used = model_rf
        if model_used is None:
            return jsonify({'error': 'Model Random Forest tidak tersedia.'}), 500
        
        # Data untuk RF (1 hari)
        features = np.array([[data['rainfall'], data['humidity'], data['elevation']]])
        
    elif model_choice == 'LSTM':
        model_used = model_lstm
        if model_used is None:
            return jsonify({'error': 'Model LSTM tidak tersedia.'}), 500
        
        # Data untuk LSTM (3 hari - harus 9 fitur)
        # Frontend mengirim list 9 nilai, kita ratakan (flatten)
        sequence_data = np.array(data['sequence_data']).flatten()
        
        # Memastikan jumlah fitur sesuai (9 fitur)
        if len(sequence_data) != 9:
             return jsonify({'error': f'Model LSTM membutuhkan 9 fitur, diterima {len(sequence_data)}.'}), 400
             
        features = sequence_data.reshape(1, -1)
        
    else:
        return jsonify({'error': 'Pilihan model tidak valid.'}), 400

    try:
        # Prediksi
        prediction_int = model_used.predict(features)[0]
        prediction_prob = model_used.predict_proba(features)

        risk_level = RISK_LABELS.get(prediction_int, 'UNKNOWN')
        highest_prob = np.max(prediction_prob) * 100

        return jsonify({
            'model': model_choice,
            'risk_level': risk_level,
            'probability': round(highest_prob, 2),
            'status': 'success'
        })
    except Exception as e:
        return jsonify({'error': f'Terjadi kesalahan saat prediksi: {str(e)}'}), 400

if __name__ == '__main__':
    print("Server Flask siap menerima permintaan di http://127.0.0.1:5000/predict")
    app.run(port=5000, debug=True)
