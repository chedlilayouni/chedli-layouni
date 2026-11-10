import pandas as pd
import numpy as np

# On fixe une graine aléatoire (seed). Cela garantit que les mêmes
# données sont recréées à chaque exécution. C'est la reproductibilité.
np.random.seed(42)

n = 500  # nombre de clients fictifs à générer

# On construit le tableau colonne par colonne.
# Chaque colonne ci-dessous est une FEATURE.
data = pd.DataFrame({
    "age": np.random.randint(18, 70, n),
    "salaire": np.random.randint(1200, 8000, n),
    "abonnement": np.random.choice(["Basique", "Standard", "Premium"], n),
    "anciennete": np.random.randint(0, 15, n),
    "consommation": np.random.randint(0, 100, n),
})

# On crée la TARGET : churn.
# Règle simplifiée : un client peu actif (consommation < 30)
# ET récent (ancienneté < 2 ans) est considéré comme "à risque".
data["churn"] = (
    (data["consommation"] < 30) &
    (data["anciennete"] < 2)
).astype(int)  # True/False est converti en 1/0

# On sauvegarde le tableau dans un fichier CSV.
data.to_csv("churn_data.csv", index=False)

# Affichages de contrôle
print(data.head())
print(f"Taux de churn : {data['churn'].mean():.1%}")