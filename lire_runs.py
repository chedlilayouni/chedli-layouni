import mlflow

# On demande à MLflow de chercher les runs et de les trier par f1 (du meilleur au pire)
runs = mlflow.search_runs(
    experiment_names=["churn-prediction"],
    order_by=["metrics.f1 DESC"]
)

# On sélectionne les colonnes à afficher
colonnes = ["tags.mlflow.runName", "params.n_estimators", "params.max_depth", "metrics.accuracy", "metrics.f1"]
print(runs[colonnes].to_string(index=False))

# Le premier de la liste est notre modèle champion
meilleur = runs.iloc[0]
print("\nMeilleur run :", meilleur["tags.mlflow.runName"], "| id :", meilleur["run_id"][:8], "...")