# ===== 1. Importer les bibliothèques =====
import os
from typing import Literal

from fastapi import FastAPI
import mlflow
import mlflow.sklearn
import pandas as pd
from pydantic import BaseModel

# ===== 2. Créer l'application =====
# Le titre s'affiche dans la documentation interactive (/docs)
app = FastAPI(title="API Churn")

# ===== 3. Lire la configuration dans les variables d'environnement =====
# Rien n'est écrit « en dur » : le même code fonctionne sur le poste,
# dans un conteneur seul, ou dans Docker Compose. Seules les variables changent.
# os.getenv("NOM", "valeur") renvoie la valeur définie ou la valeur par défaut.
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MODEL_URI = os.getenv("MODEL_URI", "models:/churn-model@champion")

# ===== 4. Charger le champion UNE SEULE FOIS, au démarrage =====
# MODEL_URI peut être une adresse du Registry (models:/...) ou un simple dossier
mlflow.set_tracking_uri(TRACKING_URI)
model = mlflow.sklearn.load_model(MODEL_URI)
print(f"Modèle chargé depuis : {MODEL_URI}")


# ===== 5. Décrire les données attendues (validées automatiquement) =====
class Client(BaseModel):
    age: int
    salaire: float
    abonnement: Literal["Basique", "Standard", "Premium"]
    anciennete: float
    consommation: float


# ===== 6. Route /health : vérifier que l'API est en vie =====
@app.get("/health")
def health():
    return {"status": "ok"}


# ===== 7. Route /predict : recevoir un client, renvoyer la prédiction =====
@app.post("/predict")
def predict(client: Client):
    # a) Transformer le client reçu en DataFrame d'une seule ligne
    ligne = pd.DataFrame([client.model_dump()])

    # b) One-hot encoding : convertir les variables catégorielles en indicatrices (0/1)
    ligne = pd.get_dummies(ligne)

    # c) Aligner les colonnes avec celles de l'entraînement (ordre identique, 0 si absente)
    ligne = ligne.reindex(columns=model.feature_names_in_, fill_value=0)

    # d) Prédire : 1 = part (churn), 0 = reste
    resultat = int(model.predict(ligne)[0])

    return {
        "churn": resultat,
        "message": "Le client va partir" if resultat == 1 else "Le client reste",
    }