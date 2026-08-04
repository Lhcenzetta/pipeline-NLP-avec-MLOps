# Pipeline NLP avec MLOps

**Conception, entrainement et supervision d'un modèle NLP de classification de tickets support à partir d'emails clients.**

Ce projet met en place une chaîne MLOps complète pour classifier automatiquement les tickets de support client en 4 catégories : `Incident`, `Request`, `Problem` et `Change`.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![MLOps](https://img.shields.io/badge/MLOps-CI%2FCD-green)

---

## Table des matières

- [Architecture](#architecture)
- [Pipeline](#pipeline)
- [Structure du projet](#structure-du-projet)
- [Installation](#installation)
- [Usage](#usage)
- [MLOps / Orchestration](#mlops--orchestration)
- [Notebooks](#notebooks)
- [Licence](#licence)

---

## Architecture

```
┌─────────────────┐   ┌─────────────────────┐   ┌──────────────────────┐
│  Emails clients │──▶│  Préprocessing NLP  │──▶│  Embeddings          │
│   (dataset)     │   │  (EDA + nettoyage)  │   │  (SentenceTransform) │
└─────────────────┘   └─────────────────────┘   └──────────┬───────────┘
                                                           │
                                             ┌─────────────┴─────────────┐
                                             ▼                           ▼
                                    ┌─────────────────┐          ┌─────────────────┐
                                    │  Classification │          │  Vector Store   │
                                    │  (LogisticReg)  │          │  (ChromaDB)     │
                                    └────────┬────────┘          └─────────────────┘
                                             ▼
                                    ┌─────────────────┐
                                    │  Prédiction     │
                                    │ Incident/Request│
                                    │ Problem/Change  │
                                    └─────────────────┘
```

## Pipeline

Le pipeline suit 4 étapes principales :

| Étape | Script | Rôle |
|-------|--------|------|
| **1. EDA & feature engineering** | `Proccessing_nlp/EDA.ipynb` | Analyse exploratoire, gestion des valeurs manquantes, création des colonnes `full_tag` et `body_cols` |
| **2. Préprocessing NLP** | `Proccessing_nlp/piplinenlp.ipynb` | Normalisation (minuscules), suppression de la ponctuation, tokenisation, suppression des stopwords |
| **3. Embeddings** | `app/embedding.py` | Encodage des textes avec `paraphrase-multilingual-MiniLM-L12-v2`, normalisation L2 |
| **4. Entrainement** | `app/piplineml.py` | Entrainement d'un classifieur `LogisticRegression`, évaluation, sauvegarde du modèle |

La prédiction (`app/main.py`) charge le modèle sauvegardé et classifie un nouveau ticket en une des 4 catégories de tickets support.

## Structure du projet

```
.
├── app/
│   ├── embedding.py        # Génération des embeddings (SentenceTransformer)
│   ├── piplineml.py        # Entrainement et évaluation du classifieur
│   ├── vector_store.py     # Indexation des embeddings dans ChromaDB
│   └── main.py             # Inference / prédiction d'un nouveau ticket
├── Proccessing_nlp/
│   ├── EDA.ipynb           # Analyse exploratoire des données
│   └── piplinenlp.ipynb    # Préprocessing NLP (nettoyage du texte)
├── model/
│   └── model.joblib        # Modèle entrainé (LogisticRegression)
├── .github/workflows/
│   └── ci-cd.yml           # Pipeline CI/CD (lint, build & push Docker)
├── dockerfile              # Image Docker (Python 3.10-slim)
├── job.yaml                # Job Kubernetes pour l'exécution du pipeline
├── requirements.txt        # Dépendances Python
└── .env                    # Variables d'environnement (chemins de données)
```

## Installation

```bash
# Cloner le dépôt
git clone <url-du-repo>
cd project1-fixxe

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt
```

### Variables d'environnement

Créez un fichier `.env` à la racine du projet et renseignez les chemins suivants :

```env
data_path = "chemin/vers/dataset.csv"           # Données brutes (emails clients)
new_data = "chemin/vers/preprocessed_data.csv"  # Données prétraitées
nlp_data = "chemin/vers/donnees_nlp.csv"        # Données après pipeline NLP
embdiding_path = "chemin/vers/embeddings.npy"   # Embeddings sauvegardés
model_path = "model/model.joblib"               # Modèle entrainé
chroma_path = "chemin/vers/chroma_db"           # Base vectorielle ChromaDB
```

> ⚠️ Les chemins par défaut dans `.env` pointent vers un répertoire local. Adaptez-les à votre environnement.

## Usage

### 1. Entrainer le modèle

```bash
python app/piplineml.py
```

Ce script charge les embeddings et les données, entraine un `LogisticRegression`, affiche le rapport de classification et sauvegarde le modèle dans `model/model.joblib`.

### 2. Faire une prédiction

```bash
python app/main.py
```

Exemple de sortie :

```
Request
```

### 3. Générer les embeddings (facultatif)

```bash
python app/embedding.py
```

### 4. Indexer dans ChromaDB (recherche de similarité)

```bash
python app/vector_store.py
```

## MLOps / Orchestration

### Docker

L'application est conteneurisée. Build de l'image :

```bash
docker build -t ml-pipeline:1.0 .
docker run --rm ml-pipeline:1.0
```

### Kubernetes

Le job Kubernetes (`job.yaml`) exécute le pipeline de machine learning en tant que job batch :

```bash
kubectl apply -f job.yaml
```

### CI/CD

Le workflow GitHub Actions (`.github/workflows/ci-cd.yml`) :

- **test** : linting `flake8`, vérification de la qualité du code (`py_compile`)
- **build-and-push** : build et push de l'image Docker vers GitHub Container Registry (`ghcr.io`) avec cache GHA
- **notify** : étape de notification à la fin du pipeline

Le workflow se déclenche sur les pushs et pull requests vers `main` et `develop`.

## Notebooks

| Notebook | Description |
|----------|-------------|
| `EDA.ipynb` | Analyse exploratoire : valeurs manquantes, fusion des tags, création de `body_cols` |
| `piplinenlp.ipynb` | Pipeline de nettoyage NLP : normalisation, ponctuation, tokenisation, stopwords |

## Licence

Ce projet est à usage pédagogique.
