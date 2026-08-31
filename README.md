# insuffisance_cardiaque_prediction

API FastAPI qui sert un modèle de machine learning entraîné pour prédire l'insuffisance cardiaque à partir de données cliniques d'un patient. Pas de README dans le dépôt à l'origine, ce document a été écrit à partir du code (`main.py`, `Dockerfile`, `requirements.txt`).

## Stack

- FastAPI + Uvicorn pour l'API
- scikit-learn / XGBoost pour l'entraînement (notebook.ipynb)
- pandas / numpy pour le traitement des données
- joblib pour charger le modèle sérialisé
- Docker pour le déploiement

## Structure du projet

```
insuffisance_cardiaque_prediction/
├── data/               → jeu de données utilisé pour l'entraînement
├── model                → modèle entraîné, sérialisé avec joblib (pas d'extension)
├── notebook.ipynb          → notebook d'entraînement / évaluation du modèle
├── main.py                  → API FastAPI qui charge le modèle et expose /predict
├── Dockerfile                 → image Docker pour servir l'API
├── requirements.txt             → dépendances Python
└── .gitignore
```

## Le modèle

Le modèle est entraîné dans `notebook.ipynb` puis exporté avec `joblib.dump(...)` dans le fichier `model`, chargé ensuite côté API avec `joblib.load("model")`.

Les 13 features attendues par le modèle sont celles du dataset classique de maladies cardiaques (Cleveland / UCI Heart Disease) :

`age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal`

## L'API

### `POST /predict`

Corps de la requête (`PatientData`) :

```json
{
  "age": 54,
  "sex": 1,
  "cp": 0,
  "trestbps": 130,
  "chol": 246,
  "fbs": 0,
  "restecg": 1,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 1.2,
  "slope": 2,
  "ca": 0,
  "thal": 2
}
```

Les 13 champs sont assemblés dans une liste `features` puis passés à `model.predict([features])`.

Réponse renvoyée :

```json
{ "prediction": "Insuffisance cardiaque détectée" }
```

ou

```json
{ "prediction": "Aucune insuffisance cardiaque détectée" }
```

Le code de calcul des probabilités (`model.predict_proba`) est présent dans le fichier mais laissé en commentaire.

## Lancer le projet

### En local

```bash
git clone https://github.com/paulcoffi/insuffisance_cardiaque_prediction.git
cd insuffisance_cardiaque_prediction

python -m venv venv
source venv/bin/activate      # venv\Scripts\activate sous Windows

pip install -r requirements.txt

uvicorn main:app --reload
```

L'API tourne sur `http://127.0.0.1:8000`, documentation Swagger sur `/docs`.

### Avec Docker

```bash
docker build -t insuffisance-cardiaque .
docker run -p 8000:8000 insuffisance-cardiaque
```

Le `Dockerfile` part de `python:3.12-slim`, installe les dépendances, copie le projet dans `/heart_failure`, expose le port 8000 et lance Uvicorn avec `--reload`.
