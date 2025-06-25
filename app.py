import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor

# Baca dataset lokal
df = pd.read_csv("datasetabalone.csv")

# Pra-pemrosesan
df['Sex'] = df['Sex'].map({'M': 0, 'F': 1, 'I': 2})
X = df.drop(columns=['Rings'])
y = df['Rings']

# Latih model
model = DecisionTreeRegressor()
model.fit(X, y)

st.title("Prediksi Umur Abalone")

# Input Manual
st.header("Prediksi Manual")
sex = st.selectbox("Sex", options=["M", "F", "I"])
length = st.number_input("Length")
diameter = st.number_input("Diameter")
height = st.number_input("Height")
whole_weight = st.number_input("Whole weight")
shucked_weight = st.number_input("Shucked weight")
viscera_weight = st.number_input("Viscera weight")
shell_weight = st.number_input("Shell weight")

if st.button("Prediksi Manual"):
    sex_num = {'M': 0, 'F': 1, 'I': 2}[sex]
    input_data = np.array([[sex_num, length, diameter, height,
                            whole_weight, shucked_weight, viscera_weight, shell_weight]])
    pred_rings = model.predict(input_data)[0]
    age = pred_rings + 1.5

    st.success(f"Prediksi jumlah cincin: {pred_rings:.2f}")
    st.info(f"Perkiraan umur: {age:.2f} tahun")

# Upload CSV untuk prediksi massal
st.header("Prediksi dari File CSV")
uploaded_file = st.file_uploader("Upload file CSV dengan kolom: Sex, Length, Diameter, Height, Whole weight, Shucked weight, Viscera weight, Shell weight", type="csv")

if uploaded_file is not None:
    user_df = pd.read_csv(uploaded_file)
    if all(col in user_df.columns for col in X.columns):
        user_df['Sex'] = user_df['Sex'].map({'M': 0, 'F': 1, 'I': 2})
        pred_rings = model.predict(user_df)
        user_df['Rings (Prediksi)'] = pred_rings
        user_df['Umur (Prediksi)'] = user_df['Rings (Prediksi)'] + 1.5
        st.subheader("Hasil Prediksi:")
        st.dataframe(user_df)

        csv_out = user_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Hasil Prediksi", csv_out, "prediksi_abalone.csv", "text/csv")
    else:
        st.error("Kolom CSV tidak sesuai. Pastikan kolomnya sama seperti: " + ', '.join(X.columns))
