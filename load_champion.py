import mlflow
import mlflow.sklearn  # Pour recharger un modèle scikit-learn
import pandas as pd

# Connexion à la même base de suivi que le TP3
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# ==========================================
# 1. Charger le modèle champion via son alias
# ==========================================
# On charge directement par l'alias sans avoir à connaître la version ou le run ID exact
model = mlflow.sklearn.load_model("models:/churn-model@champion")

print("Modèle chargé :", type(model).__name__)
print("Variables attendues :", list(model.feature_names_in_))


# ==========================================
# 2. Créer un nouveau client (données de test)
# ==========================================
# Les colonnes d'abonnement sont déjà transformées en 0/1 (ici, abonnement Standard)
client = pd.DataFrame([{
    "age": 35,
    "salaire": 2500,
    "anciennete": 1,
    "consommation": 10,
    "abonnement_Premium": False,
    "abonnement_Standard": True,
}])


# ==========================================
# 3. Aligner les colonnes avec le modèle
# ==========================================
# On garde uniquement les colonnes connues du modèle et dans le bon ordre
client = client[list(model.feature_names_in_)]


# ==========================================
# 4. Effectuer la prédiction
# ==========================================
# 1 = le client va partir (churn), 0 = il reste
prediction = model.predict(client)[0]
resultat_texte = "OUI" if prediction == 1 else "NON"

print("Le client va-t-il partir ? ->", resultat_texte)