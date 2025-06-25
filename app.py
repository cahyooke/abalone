import streamlit as st
import numpy as np
import joblib

model = joblib.load("decision_tree_model.pkl")

st.title("Prediksi Umur Abalone (Decision Tree)")
st.markdown("Masukkan fitur-fitur abalone:")

sex = st.selectbox("Sex", options=["M", "F", "I"])

sex_map = {"M": 0, "F": 1, "I": 2}
sex_num = sex_map[sex]

length = st.number_input("Length")
diameter = st.number_input("Diameter")
height = st.number_input("Height")
whole_weight = st.number_input("Whole weight")
shucked_weight = st.number_input("Shucked weight")
viscera_weight = st.number_input("Viscera weight")
shell_weight = st.number_input("Shell weight")

sex_map = {"_
