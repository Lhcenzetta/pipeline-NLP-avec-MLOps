# NLP Pipeline with MLOps

**Design, training, and monitoring of an NLP model for classifying support tickets from customer emails.**

This project implements a full MLOps pipeline to automatically classify customer support tickets into 4 categories: `Incident`, `Request`, `Problem`, and `Change`.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![MLOps](https://img.shields.io/badge/MLOps-CI%2FCD-green)

---

## Table of Contents

- [Architecture](#architecture)
- [Pipeline](#pipeline)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [MLOps / Orchestration](#mlops--orchestration)
- [Notebooks](#notebooks)
- [License](#license)

---

## Architecture

```
┌─────────────────┐   ┌─────────────────────┐   ┌──────────────────────┐
│  Customer emails│──▶│  NLP Preprocessing  │──▶│  Embeddings          │
│   (dataset)     │   │  (EDA + cleaning)   │   │  (SentenceTransform) │
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
                                    │  Prediction     │
                                    │ Incident/Request│
                                    │ Problem/Change  │
                                    └─────────────────┘
```

## Pipeline

The pipeline follows 4 main steps:

| Step | Script | Role |
|------|--------|------|
| **1. EDA & feature engineering** | `Proccessing_nlp/EDA.ipynb` | Exploratory analysis, missing value handling, creation of `full_tag` and `body_cols` columns |
| **2. NLP preprocessing** | `Proccessing_nlp/piplinenlp.ipynb` | Normalization (lowercase), punctuation removal, tokenization, stopword removal |
| **3. Embeddings** | `app/embedding.py` | Text encoding with `paraphrase-multilingual-MiniLM-L12-v2`, L2 normalization |
| **4. Training** | `app/piplineml.py` | `LogisticRegression` classifier training, evaluation, model saving |

Prediction (`app/main.py`) loads the saved model and classifies a new ticket into one of the 4 support ticket categories.

## Project Structure

```
.
├── app/
│   ├── embedding.py        # Embedding generation (SentenceTransformer)
│   ├── piplineml.py        # Classifier training and evaluation
│   ├── vector_store.py     # Embedding indexing into ChromaDB
│   └── main.py             # Inference / prediction of a new ticket
├── Proccessing_nlp/
│   ├── EDA.ipynb           # Exploratory data analysis
│   └── piplinenlp.ipynb    # NLP preprocessing (text cleaning)
├── model/
│   └── model.joblib        # Trained model (LogisticRegression)
├── .github/workflows/
│   └── ci-cd.yml           # CI/CD pipeline (lint, Docker build & push)
├── dockerfile              # Docker image (Python 3.10-slim)
├── job.yaml                # Kubernetes job for pipeline execution
├── requirements.txt        # Python dependencies
└── .env                    # Environment variables (data paths)
```

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd project1-fixxe

# Create a virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file at the project root and fill in the following paths:

```env
data_path = "path/to/dataset.csv"               # Raw data (customer emails)
new_data = "path/to/preprocessed_data.csv"      # Preprocessed data
nlp_data = "path/to/nlp_data.csv"               # Data after NLP pipeline
embdiding_path = "path/to/embeddings.npy"       # Saved embeddings
model_path = "model/model.joblib"               # Trained model
chroma_path = "path/to/chroma_db"               # ChromaDB vector store
```

> ⚠️ The default paths in `.env` point to a local directory. Adjust them to your environment.

## Usage

### 1. Train the model

```bash
python app/piplineml.py
```

This script loads the embeddings and data, trains a `LogisticRegression`, prints the classification report, and saves the model to `model/model.joblib`.

### 2. Make a prediction

```bash
python app/main.py
```

Example output:

```
Request
```

### 3. Generate embeddings (optional)

```bash
python app/embedding.py
```

### 4. Index into ChromaDB (similarity search)

```bash
python app/vector_store.py
```

## MLOps / Orchestration

### Docker

The application is containerized. Build the image:

```bash
docker build -t ml-pipeline:1.0 .
docker run --rm ml-pipeline:1.0
```

### Kubernetes

The Kubernetes job (`job.yaml`) runs the machine learning pipeline as a batch job:

```bash
kubectl apply -f job.yaml
```

### CI/CD

The GitHub Actions workflow (`.github/workflows/ci-cd.yml`):

- **test** : `flake8` linting, code quality check (`py_compile`)
- **build-and-push** : Docker image build and push to GitHub Container Registry (`ghcr.io`) with GHA cache
- **notify** : notification step at the end of the pipeline

The workflow triggers on pushes and pull requests to `main` and `develop`.

## Notebooks

| Notebook | Description |
|----------|-------------|
| `EDA.ipynb` | Exploratory analysis: missing values, tag merging, `body_cols` creation |
| `piplinenlp.ipynb` | NLP cleaning pipeline: normalization, punctuation, tokenization, stopwords |

## License

This project is for educational purposes.
