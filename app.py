import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from ucimlrepo import fetch_ucirepo

# Load dataset dari UCI untuk melatih model
abalone = fetch_ucirepo(id=1)
X = abalone.data.features
y = abalone.data.targets
X['Sex'] = X['Sex'].map({'M': 0, 'F': 1, 'I': 2})

# Latih model
model = DecisionTreeRegressor()
model.fit(X, y)

st.title("Prediksi Umur Abalone")

# Fitur 1: Input manual
st.header("Input Manual")
sex = st.selectbox("Sex", options=["M", "F", "I"])
length = st.number_input("Length")
diameter = st.number_input("Diameter")
height = st.number_input("Height")
whole_weight = st.number_input("Whole weight")
shucked_weight = st.number_input("Shucked weight")
viscera_weight = st.number_input("Viscera weight")
shell_weight = st.number_input("Shell weight")

sex_map = {"M": 0, "F": 1, "I": 2}
sex_num = sex_map[sex]

if st.button("Prediksi (Manual)"):
    x = np.array([[sex_num, length, diameter, height,
                   whole_weight, shucked_weight, viscera_weight, shell_weight]])
    pred = model.predict(x)[0]
    umur = pred + 1.5
    st.success(f"Perkiraan jumlah cincin: {pred:.2f}")
    st.info(f"Perkiraan umur abalone: {umur:.2f} tahun")

# Fitur 2: Input dari CSV
st.header("Upload File CSV")
uploaded_file = st.file_uploader("Upload CSV dengan kolom: Sex, Length, Diameter, Height, Whole weight, Shucked weight, Viscera weight, Shell weight", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Cek kolom
    expected_cols = ['Sex', 'Length', 'Diameter', 'Height',
                     'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight']
    if all(col in df.columns for col in expected_cols):
        df['Sex'] = df['Sex'].map(sex_map)
        predictions = model.predict(df)
        df['Rings (Prediksi)'] = predictions
        df['Umur (Prediksi)'] = df['Rings (Prediksi)'] + 1.5
        st.subheader("Hasil Prediksi:")
        st.dataframe(df)
        csv_result = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Hasil Prediksi", csv_result, "prediksi_abalone.csv", "text/csv")
    else:
        st.error("Kolom CSV tidak sesuai format yang diharapkan.")
