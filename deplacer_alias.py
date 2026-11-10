import mlflow
from mlflow import MlflowClient

# Connexion à la base de suivi MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()


# ==========================================
# 1. Retrouver le run « essai-2 »
# ==========================================
# Recherche du run spécifique correspondant au modèle le plus faible du TP3
runs = mlflow.search_runs(
    experiment_names=["churn-prediction"],
    filter_string="tags.mlflow.runName = 'essai-2'"
)
run_id = runs.iloc[0]["run_id"]


# ==========================================
# 2. Enregistrer le run comme nouvelle version
# ==========================================
# Cela crée une NOUVELLE VERSION du modèle 'churn-model' dans le Registry
v = mlflow.register_model(f"runs:/{run_id}/model", "churn-model")

print("Nouvelle version créée :", v.version)


# ==========================================
# 3. Vérifier où pointe l'alias « champion »
# ==========================================
actuel = client.get_model_version_by_alias("churn-model", "champion")

print("Champion actuel : version", actuel.version)


# ==========================================
# 4. Déplacer l'alias, puis le remettre
# ==========================================
# Déplacement temporaire de l'alias vers la nouvelle version (v)
client.set_registered_model_alias("churn-model", "champion", v.version)
print(
    "Après déplacement : champion -> version",
    client.get_model_version_by_alias("churn-model", "champion").version
)

# Remise de l'alias sur la version d'origine (actuel)
client.set_registered_model_alias("churn-model", "champion", actuel.version)
print(
    "Après retour : champion -> version",
    client.get_model_version_by_alias("churn-model", "champion").version
)