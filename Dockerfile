# 1. Image de base
FROM python:3.11-slim

# 2. Dossier de travail
WORKDIR /app

# 3. Installer les dépendances (mise en cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copier le code et le dossier model
COPY app.py .
COPY model/ model/

# 5. Variable par défaut
ENV MODEL_URI=model

# 6. Port documenté
EXPOSE 8000

# 7. Démarrage de l'API
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]