import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib

from dotenv import load_dotenv
import os

load_dotenv()
EMBEDDING_PATH =  os.getenv("embdiding_path")
path_data = os.getenv("nlp_data")

df = pd.read_csv(path_data)
embeddings = np.load(EMBEDDING_PATH)

X = embeddings
ls = LabelEncoder()

y = ls.fit_transform(df["type"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))

joblib.dump(model, "model.joblib")
