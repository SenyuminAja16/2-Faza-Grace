import numpy as np
import pickle
import streamlit as st
import os
from scipy.spatial.distance import cdist

base_path = os.path.dirname(__file__)
file_path = os.path.join(base_path, "agg_clustering_Kelompok2.pkl")
centroid_path = os.path.join(base_path, "centroids.npy")
scaler_path = os.path.join(base_path, "scaler.pkl")
encoder_path = os.path.join(base_path, "encoder.pkl")

if not os.path.exists(file_path) or not os.path.exists(centroid_path):
    st.error("File model atau centroid tidak ditemukan.")
    st.stop()

with open(file_path, 'rb') as f:
    model = pickle.load(f)

centroids = np.load(centroid_path)

with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

with open(encoder_path, "rb") as f:
    encoder = pickle.load(f)

st.title('Clustering Zona Berdasarkan Polutan dan Cuaca')

col1, col2 = st.columns(2)

with col1:
    carbon_monoxide = st.number_input('Input kadar Karbon Monoxide (ppb)')
    
with col2:
    pm25 = st.number_input('Input kadar PM 2.5 (µg/m³)')

with col1:
    quality_ozone = st.number_input('Input kadar Ozone (ppb)')
    
with col2:
    pm10 = st.number_input('Input kadar PM 10 (µg/m³)')

with col1:
    shulphur_dioxide = st.number_input('Input kadar Shulphur Dioxide (ppb)')

with col2:
    wind_kph = st.number_input('Input Rata Rata Kecepatan Angin (km/h)')

with col1:
    nitrogen_dioxide = st.number_input('Input kadar Nitrogen Dioxide (ppb)')

with col2:
    humidity = st.number_input('Input Rata Rata Kelembapan (%)')

temperature = st.number_input('Input Rata Rata Suhu (C°)')

pm_ratio = pm25 / (pm10 + 1e-5)

def categorize_wind(speed):
    if speed < 5:
        return "Low"
    elif 5 <= speed <= 15:
        return "Medium"
    else:
        return "High"

wind_condition = encoder.transform(np.array([[categorize_wind(wind_kph)]]))[0][0]
humidity_temperature_ratio = humidity / (temperature + 1e-5)
dew_point = temperature - ((100 - humidity) / 5)

input_data = np.array([[carbon_monoxide, quality_ozone, shulphur_dioxide,
                        nitrogen_dioxide, pm25, pm10, pm_ratio,
                        humidity_temperature_ratio, dew_point]])

scaled_data = scaler.transform(input_data)

index_wind_condition = 7  
input_ready = np.insert(scaled_data, index_wind_condition, wind_condition, axis=1)

zona_cluster = ''
if st.button('Test'):
    input_ready = np.array(input_ready).reshape(1, -1)
    
    distances = cdist(input_ready, centroids, metric="euclidean")
    zona = np.argmin(distances, axis=1)[0]
    
    zona_mapping = {
        0: "Zona 1: Kualitas udara baik dengan kadar polutan rendah (CO, NO₂, PM2.5). Rasio kelembapan terhadap suhu tinggi, serta titik embun lebih rendah, menandakan kondisi lingkungan yang lebih sehat.",
        1: "Zona 2: Konsentrasi SO₂ dan NO₂ sangat tinggi, serta peningkatan signifikan pada PM2.5 dan PM10. Indikasi bahwa area ini terdampak oleh aktivitas industri dan transportasi, sehingga perlu pengelolaan emisi yang lebih ketat.",
        2: "Zona 3: Ozon lebih rendah dibanding cluster lain, tetapi terdapat peningkatan kadar PM2.5. Kemungkinan besar berasal dari lalu lintas kendaraan atau sumber polusi skala menengah.",
        3: "Zona 4: Konsentrasi PM2.5 dan PM10 sangat tinggi, disertai dengan rasio kelembapan terhadap suhu yang juga meningkat. Area ini memiliki risiko kesehatan yang serius, terutama bagi kelompok rentan, sehingga memerlukan mitigasi segera"
    }
    zona_cluster = zona_mapping.get(zona, "Zona tidak dikenal")

    st.success(f"Zona ini masuk ke dalam {zona_cluster}")
