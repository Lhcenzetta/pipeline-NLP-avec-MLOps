import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from dotenv import load_dotenv
import os

load_dotenv()

data_path = os.getenv("nlp_data")


df = pd.read_csv(data_path)

texts = df["body_cols"].astype(str).tolist()

def embding_text(text):
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    embeddings = model.encode(
    text,
    batch_size=32,
    show_progress_bar=True
    )
    if len(embeddings.shape) == 1:
        embeddings = embeddings.reshape(1, -1)
    embeddings = normalize(embeddings, norm="l2")
    return embeddings

embeddings = embding_text(texts)

print(embeddings)

np.save("data/embeddings1.npy", embeddings)


