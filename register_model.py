# ==========================================
# 1. Importer les bibliothèques
# ==========================================
import mlflow  # Le module principal de MLflow
from mlflow import MlflowClient  # Le « client » : permet de gérer le Registry


# ==========================================
# 2. Se connecter à la base de suivi
# ==========================================
# Connexion au fichier mlflow.db du dossier courant (contient tous les runs)
mlflow.set_tracking_uri("sqlite:///mlflow.db")


# ==========================================
# 3. Retrouver le meilleur run
# ==========================================
# Recherche des runs dans l'expérience 'churn-prediction', triés par F1-score décroissant
runs = mlflow.search_runs(
    experiment_names=["churn-prediction"],
    order_by=["metrics.f1 DESC"]
)

best = runs.iloc[0]          # Première ligne = le meilleur run
run_id = best["run_id"]      # Récupération de l'identifiant unique

print(f"Meilleur run : {best['tags.mlflow.runName']} | ID : {run_id[:8]} ...")


# ==========================================
# 4. Enregistrer le modèle dans le Registry
# ==========================================
# "runs:/<id>/model" = le dossier 'model' du run sélectionné
# "churn-model" = le nom du modèle dans le catalogue (créé automatiquement s'il n'existe pas)
result = mlflow.register_model(f"runs:/{run_id}/model", "churn-model")

print(f"Modèle enregistré : churn-model | Version : {result.version}")


# ==========================================
# 5. Poser l'alias « champion » sur la version
# ==========================================
client = MlflowClient()
client.set_registered_model_alias("churn-model", "champion", result.version)

print(f"Alias 'champion' attribué à la version : {result.version}")