import mlflow
import mlflow.sklearn
import pandas as pd

# 1. Retrouver le meilleur run (comme dans lire_runs.py)
runs = mlflow.search_runs(experiment_names=["churn-prediction"], order_by=["metrics.f1 DESC"])
run_id = runs.iloc[0]["run_id"]

# 2. Recharger le modèle directement depuis MLflow
print(f"Chargement du modèle depuis le run : {run_id[:8]}...")
model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

print("Modèle rechargé :", type(model).__name__)
print("Nombre de variables attendues :", model.n_features_in_)

# 3. Créer un nouveau client (profil à risque : jeune, faible conso, récent)
client = pd.DataFrame([{
    "age": 35, 
    "salaire": 2500, 
    "anciennete": 1, 
    "consommation": 10,
    "abonnement_Premium": False, 
    "abonnement_Standard": True,
}])

# 4. S'assurer que le client a les mêmes colonnes que celles vues par le modèle à l'entraînement
client = client[list(model.feature_names_in_)]

# 5. Faire la prédiction (1 = part, 0 = reste)
prediction = model.predict(client)[0]
print("Le client va-t-il partir ? ->", "OUI" if prediction == 1 else "NON")