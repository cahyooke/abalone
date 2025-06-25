import streamlit as st
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from ucimlrepo import fetch_ucirepo

# Load dataset dari UCI
abalone = fetch_ucirepo(id=1)
X = abalone.data.features
y = abalone.data.targets

# Konversi kategori ke angka
X['Sex'] = X['Sex'].map({'M': 0, 'F': 1, 'I': 2})

# Latih model langsung
model = DecisionTreeRegressor()
model.fit(X, y)

st.title("Prediksi Umur Abalone")
st.markdown("Masukkan fitur-fitur abalone:")

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

if st.button("Prediksi"):
    x = np.array([[sex_num, length, diameter, height,
                   whole_weight, shucked_weight, viscera_weight, shell_weight]])
    pred = model.predict(x)[0]
    umur = pred + 1.5

    st.success(f"Perkiraan jumlah cincin abalone: {pred:.2f}")
    st.info(f"Perkiraan umur abalone: {umur:.2f} tahun")
