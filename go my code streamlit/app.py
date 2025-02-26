import streamlit as st
import joblib
import numpy as np

# Charger le modèle KNN
model = joblib.load("knn_model.pkl")

# Titre de l'application
st.title("Prédiction d'utilisation d'un compte bancaire")

# Création des champs de saisie pour les fonctionnalités (uniquement des valeurs numériques)
country = st.number_input("Pays (kenya)", min_value=0, value=1)
location_type = st.number_input("Type de lieu (0 = Rural, 1 = Urbain)", min_value=0, max_value=1, value=0)
cellphone_access = st.number_input("Accès au téléphone portable (0 = Non, 1 = Oui)", min_value=0, max_value=1, value=1)
household_size = st.number_input("Taille du foyer", min_value=1, value=3)
age_of_respondent = st.number_input("Âge du répondant", min_value=18, max_value=100, value=30)
gender_of_respondent = st.number_input("Genre (0 = Femme, 1 = Homme)", min_value=0, max_value=1, value=1)
relationship_with_head = st.number_input("Lien avec le chef de famille (code numérique)", min_value=0, value=1)
marital_status = st.number_input("Statut marital (code numérique)", min_value=0, value=1)
education_level = st.number_input("Niveau d'éducation (code numérique)", min_value=0, value=1)
job_type = st.number_input("Type d'emploi (code numérique)", min_value=0, value=1)

# Bouton de validation
if st.button("Prédire"):
    # Créer un tableau numpy des valeurs saisies
    input_data = np.array([[country, location_type, cellphone_access, household_size, 
                            age_of_respondent, gender_of_respondent, relationship_with_head, 
                            marital_status, education_level, job_type]])

    # Prédire avec le modèle KNN
    prediction = model.predict(input_data)

    # Afficher le résultat
    result = "Possède un compte bancaire" if prediction[0] == 1 else "Ne possède PAS de compte bancaire"
    st.success(f"Résultat de la prédiction : {result}")