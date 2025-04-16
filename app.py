import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image
import matplotlib.pyplot as plt
import io

# Configuration de la page
st.set_page_config(page_title="Prédiction du prix de vente d'une voiture", page_icon="🚗", layout="centered")

# Personnalisation CSS
st.markdown(
    """
    <style>
    .main {
        background-color: #800000;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Chargement du logo ENISE
st.image("logoenise.png", width=200)

st.title("💸 Prédiction du prix de vente d'une voiture")

# Chargement du modèle
model = joblib.load("modele_voiture.pkl")

# Initialisation de l'historique des prédictions
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=[
        "Present_Price", "Kms_Driven", "Car_Age",
        "Fuel_Type", "Seller_Type", "Transmission", "Owner", "Prix prédit", "Perte de valeur"
    ])

# Formulaire de saisie
with st.form("prediction_form"):
    st.subheader("📝 Informations sur la voiture")
    present_price = st.number_input("Prix neuf de la voiture (en k €)", min_value=0.0, step=0.1)
    kms_driven = st.number_input("Kilométrage (Kms Driven)", min_value=0)
    car_age = st.number_input("Âge de la voiture (en années)", min_value=0)
    fuel_type = st.selectbox("Type de carburant", ["Petrol", "Diesel", "CNG"])
    seller_type = st.selectbox("Type de vendeur", ["Dealer", "Individual"])
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
    owner = st.selectbox("Nombre de propriétaires précédents", [0, 1, 3])
    submit = st.form_submit_button("Prédire le prix")

if submit:
    # Création du DataFrame d'entrée
    input_data = pd.DataFrame({
        "Present_Price": [present_price],
        "Kms_Driven": [kms_driven],
        "Car_Age": [car_age],
        "Fuel_Type": [fuel_type],
        "Seller_Type": [seller_type],
        "Transmission": [transmission],
        "Owner": [owner]
    })

    # Prédiction
    prediction = model.predict(input_data)[0]
    depreciation = present_price - prediction

    st.success(f"✅ Prix estimé : {prediction:.2f}k €")
    st.info(f"📉 Perte estimée de valeur : {depreciation:.2f}k €")

    # Mise à jour de l'historique
    input_data["Prix prédit"] = prediction
    input_data["Perte de valeur"] = depreciation
    st.session_state.history = pd.concat([st.session_state.history, input_data], ignore_index=True)

# Affichage de l'historique
if not st.session_state.history.empty:
    st.subheader("📚 Historique des prédictions")
    st.dataframe(st.session_state.history)

    # Téléchargement de l'historique
    csv = st.session_state.history.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Télécharger l'historique au format CSV",
        data=csv,
        file_name='historique_predictions.csv',
        mime='text/csv'
    )


