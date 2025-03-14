import numpy as np
import pickle
import streamlit as st
import os
from scipy.spatial.distance import cdist

# Load Model
file_path = r'C:\Belajar Data\Intern BCC\2-Faza-Grace\Deploy\agg_clustering_Kelompok2.pkl'
centroid_path = r"C:\Belajar Data\Intern BCC\2-Faza-Grace\Deploy\centroids.npy"
scaler_path = r"C:\Belajar Data\Intern BCC\2-Faza-Grace\Deploy\scaler.pkl"
encoder_path = r"C:\Belajar Data\Intern BCC\2-Faza-Grace\Deploy\encoder.pkl"

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

# UI
st.title('Clustering Zona Berdasarkan Polutan dan Cuaca')

carbon_monoxide = st.number_input('Input kadar Karbon Monoxide')
quality_ozone = st.number_input('Input kadar Ozone')
shulphur_dioxide = st.number_input('Input kadar Shulphur Dioxide')
nitrogen_dioxide = st.number_input('Input kadar Nitrogen Dioxide')
pm25 = st.number_input('Input kadar PM 2.5')
pm10 = st.number_input('Input kadar PM 10')
wind_kph = st.number_input('Input Rata Rata Kecepatan Angin')
humidity = st.number_input('Input Rata Rata Kelembapan')
temperature = st.number_input('Input Rata Rata Suhu')

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
    
    # Menentukan cluster berdasarkan jarak ke centroid
    distances = cdist(input_ready, centroids, metric="euclidean")
    zona = np.argmin(distances, axis=1)[0]
    
    zona_mapping = {
        0: "Zona 0: Udara Sangat Bersih",
        1: "Zona 1: Udara Cukup Bersih",
        2: "Zona 2: Udara Tercemar",
        3: "Zona 3: Udara Sangat Tercemar"
    }
    zona_cluster = zona_mapping.get(zona, "Zona tidak dikenal")

    st.success(f"Zona ini masuk ke dalam {zona_cluster}")
