import chromadb
import numpy as np
import pandas as pd
from tqdm import tqdm

from dotenv import load_dotenv
import os

load_dotenv()
EMBEDDING_PATH =  os.getenv("embdiding_path")
DATA_PATH = os.getenv("nlp_data")
CHROMA_PATH = os.getenv("chroma_path")


df = pd.read_csv(DATA_PATH)
embeddings = np.load(EMBEDDING_PATH)


client = chromadb.PersistentClient(path=CHROMA_PATH)

collection = client.get_or_create_collection(
    name="support_tickets"
)


ids = [str(i) for i in df.index]
documents = df["body_cols"].astype(str).tolist()
metadatas = df[["type", "priority"]].to_dict(orient="records")

batch_size = 5000

for i in tqdm(range(0, len(ids), batch_size)):
    collection.add(
        ids=ids[i:i+batch_size],
        embeddings=embeddings[i:i+batch_size].tolist(),
        documents=documents[i:i+batch_size],
        metadatas=metadatas[i:i+batch_size]
    )

def get_similare(embedding):
    results = collection.query(
    query_embeddings=embedding.tolist(),
    n_results=3 
)
    return results
