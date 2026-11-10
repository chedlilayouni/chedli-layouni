import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from sklearn.ensemble import RandomForestClassifier

# 1. Pointer vers la base SQLite locale
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("churn-tp4")

# 2. Charger les données
df = pd.read_csv("churn_data.csv")
X = pd.get_dummies(df.drop(columns=["churn"]), columns=["abonnement"])
y = df["churn"]

# 3. Entraîner le modèle
modele = RandomForestClassifier(n_estimators=100, random_state=42)
modele.fit(X, y)

# 4. Enregistrer dans le Registry
with mlflow.start_run():
    mlflow.sklearn.log_model(
        modele,
        artifact_path="model",
        registered_model_name="churn-model",
        serialization_format="cloudpickle",
    )

# 5. Poser l'alias champion
client = MlflowClient()
version = client.search_model_versions("name='churn-model'")[0].version
client.set_registered_model_alias("churn-model", "champion", version)
print(f"Modèle churn-model version {version} créé avec l'alias champion !")