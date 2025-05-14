from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

MODEL_PATH = r'D:\Adeeee\New folder\Dami\HousePriceProyek\stacked_model.pkl'
loaded_model = joblib.load(MODEL_PATH)  # loaded_model

# Fungsi prediksi harga rumah
def predict_house_price(features):
    feature_names = [
        'OverallQual', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF',
        '1stFlrSF', 'GrLivArea', 'FullBath', 'TotRmsAbvGrd',
        'GarageCars', 'GarageArea'
    ]

    # Load model & scaler dari dictionary
    meta_model = loaded_model['meta_model']
    base_models = loaded_model['base_models']
    scaler = loaded_model['scaler']

    # Konversi input ke DataFrame
    features_df = pd.DataFrame([features], columns=feature_names)

    # Scaling
    features_scaled = scaler.transform(features_df)

    # Prediksi base models
    meta_features = np.column_stack([
        base_models[m].predict(features_scaled) for m in base_models
    ])

    # Prediksi akhir
    final_prediction = meta_model.predict(meta_features)[0]
    return np.expm1(final_prediction)  # Transformasi balik log1p

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Ambil input dari form (tanpa GarageYrBlt)
        overall_quality = int(request.form['overall_quality'])
        year_built = int(request.form['year_built'])
        year_remod_add = int(request.form['year_remod_add'])
        total_bsmt_sf = int(request.form['total_bsmt_sf'])
        first_flr_sf = int(request.form['first_flr_sf'])
        gr_liv_area = int(request.form['gr_liv_area'])
        full_bath = int(request.form['full_bath'])
        total_rooms_abv_grd = int(request.form['total_rooms_abv_grd'])
        garage_cars = int(request.form['garage_cars'])
        garage_area = int(request.form['garage_area'])

        # Susun dictionary fitur
        features = {
            'OverallQual': overall_quality,
            'YearBuilt': year_built,
            'YearRemodAdd': year_remod_add,
            'TotalBsmtSF': total_bsmt_sf,
            '1stFlrSF': first_flr_sf,
            'GrLivArea': gr_liv_area,
            'FullBath': full_bath,
            'TotRmsAbvGrd': total_rooms_abv_grd,
            'GarageCars': garage_cars,
            'GarageArea': garage_area
        }

        predicted_price = predict_house_price(features)
        return render_template('index.html', predicted_price=predicted_price)

#standar untuk memulai aplikasi web di Flask.
if __name__ == '__main__':
    app.run(debug=True)
