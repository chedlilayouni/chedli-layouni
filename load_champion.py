import mlflow
import mlflow.sklearn
import pandas as pd

# Connexion à la base de suivi locale SQLite
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# ==========================================
# 1. Charger le modèle champion via son alias
# ==========================================
model = mlflow.sklearn.load_model("models:/churn-model@champion")

print("Modèle chargé :", type(model).__name__)
print("Variables attendues :", list(model.feature_names_in_))


# ==========================================
# 2. Créer un nouveau client (données de test)
# ==========================================
client = pd.DataFrame(
    [
        {
            "age": 35,
            "salaire": 2500,
            "anciennete": 1,
            "consommation": 10,
            "abonnement_Premium": False,
            "abonnement_Standard": True,
        }
    ]
)


# ==========================================
# 3. Aligner les colonnes avec le modèle
# ==========================================
# reindex garantit le bon ordre et remplit à 0 les colonnes absentes
client = client.reindex(columns=model.feature_names_in_, fill_value=0)


# ==========================================
# 4. Effectuer la prédiction
# ==========================================
# 1 = le client va partir (churn), 0 = il reste
prediction = int(model.predict(client)[0])
resultat_texte = "OUI" if prediction == 1 else "NON"

print("Le client va-t-il partir ? ->", resultat_texte)