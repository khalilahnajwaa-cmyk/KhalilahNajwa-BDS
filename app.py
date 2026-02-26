import streamlit as st
import pandas as pd
import pickle
import numpy as np

# --- 1. Load the trained model and scaler ---
@st.cache_resource
def load_model_and_scaler():
    try:
        with open('gradient_boosting_model.pkl', 'rb') as file:
            model_gb = pickle.load(file)
        with open('standard_scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
        return model_gb, scaler
    except FileNotFoundError:
        st.error("Error: Model or scaler files not found. Please ensure 'gradient_boosting_model.pkl' and 'standard_scaler.pkl' are in the same directory.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model or scaler: {e}")
        st.stop()

model_gb, scaler = load_model_and_scaler()

# --- 2. Define Mappings and Expected Feature Order (from training) ---
pendidikan_mapping = {'D3': 0, 'S1': 1, 'SMA': 2, 'SMK': 3}
jurusan_mapping = {'administrasi': 0, 'desain grafis': 1, 'otomotif': 2, 'teknik las': 3, 'teknik listrik': 4}

one_hot_categorical_columns = [
    'Jenis_Kelamin_Laki-laki', 'Jenis_Kelamin_P', 'Jenis_Kelamin_Wanita',
    'Status_Bekerja_Belum Bekerja', 'Status_Bekerja_Sudah Bekerja'
]

# This order must EXACTLY match the feature order used during model training
expected_feature_order = [
    'Pendidikan', 'Jurusan', 'Jenis_Kelamin_Laki-laki', 'Jenis_Kelamin_P', 'Jenis_Kelamin_Wanita',
    'Status_Bekerja_Belum Bekerja', 'Status_Bekerja_Sudah Bekerja',
    'Usia', 'Durasi_Jam', 'Nilai_Ujian'
]

# --- 3. Streamlit UI for User Input ---
st.title("Prediksi Gaji Pertama Peserta Vokasi")
st.write("Aplikasi ini memprediksi gaji pertama berdasarkan data peserta pelatihan vokasi.")
st.markdown("--- ")

st.header("Input Data Peserta")

# Numerical Inputs
usia = st.slider("Usia (tahun)", min_value=18, max_value=59, value=31)
durasi_jam = st.slider("Durasi Pelatihan (jam)", min_value=20, max_value=99, value=58)
nilai_ujian = st.slider("Nilai Ujian", min_value=55.0, max_value=97.5, value=75.0, step=0.1)

# Categorical Inputs
jenis_kelamin = st.selectbox("Jenis Kelamin", ['Laki-laki', 'P', 'Wanita'])
pendidikan = st.selectbox("Pendidikan", ['SMA', 'SMK', 'D3', 'S1'])
jurusan = st.selectbox("Jurusan", ['administrasi', 'desain grafis', 'otomotif', 'teknik las', 'teknik listrik'])
status_bekerja = st.selectbox("Status Bekerja", ['Sudah Bekerja', 'Belum Bekerja'])

# --- 4. Preprocessing User Input ---
if st.button("Prediksi Gaji"):
    # Create a DataFrame from user inputs
    input_data = pd.DataFrame({
        'Jenis_Kelamin': [jenis_kelamin],
        'Usia': [usia],
        'Pendidikan': [pendidikan],
        'Jurusan': [jurusan],
        'Durasi_Jam': [durasi_jam],
        'Nilai_Ujian': [nilai_ujian],
        'Status_Bekerja': [status_bekerja]
    })

    # Apply cleaning steps consistent with training data
    input_data['Jenis_Kelamin'] = input_data['Jenis_Kelamin'].replace({'Pria': 'Laki-laki', 'L': 'Laki-laki'})
    input_data['Jurusan'] = input_data['Jurusan'].str.lower()

    # Apply Label Encoding
    input_data['Pendidikan'] = input_data['Pendidikan'].map(pendidikan_mapping)
    input_data['Jurusan'] = input_data['Jurusan'].map(jurusan_mapping)

    # Handle potential NaNs from mapping (if user selected unmapped value, though unlikely with selectbox)
    for col in ['Pendidikan', 'Jurusan']:
        if input_data[col].isnull().any():
            st.warning(f"Warning: '{col}' input could not be mapped. This might affect prediction accuracy.")

    # Apply One-Hot Encoding
    one_hot_temp_df = pd.get_dummies(input_data[['Jenis_Kelamin', 'Status_Bekerja']], prefix=['Jenis_Kelamin', 'Status_Bekerja'])
    one_hot_temp_df = one_hot_temp_df.astype(int)

    # Reindex the one-hot encoded columns to match the training data's exact columns and order
    final_one_hot_df = pd.DataFrame(0, index=input_data.index, columns=one_hot_categorical_columns) # Initialize with all expected OHE columns
    for col in one_hot_categorical_columns:
        if col in one_hot_temp_df.columns:
            final_one_hot_df[col] = one_hot_temp_df[col]

    # Combine all preprocessed features
    processed_features = pd.concat([
        input_data[['Pendidikan', 'Jurusan', 'Usia', 'Durasi_Jam', 'Nilai_Ujian']],
        final_one_hot_df
    ], axis=1)

    # Ensure the order of columns matches the training data (critical for prediction)
    processed_features = processed_features[expected_feature_order]

    # Apply Scaling
    input_scaled = scaler.transform(processed_features)
    input_scaled_df = pd.DataFrame(input_scaled, columns=expected_feature_order)

    # --- 5. Make Prediction ---
    predicted_salary = model_gb.predict(input_scaled_df)

    # --- 6. Display Results ---
    st.markdown("--- ")
    st.subheader("Hasil Prediksi")
    st.success(f"Prediksi Gaji Pertama: **Rp {predicted_salary[0]:,.2f} Juta**")
    st.info("Disclaimer: Prediksi ini didasarkan pada model yang dilatih dan data yang tersedia. Hasil sebenarnya dapat bervariasi.")
