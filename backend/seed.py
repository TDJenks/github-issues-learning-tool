from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import umap
from sklearn.preprocessing import MinMaxScaler
from db import get_conn, init_db, reset_db

#-------------------------------------------------------------------
# Currently this is hardcoded to work specifically for github issues
#-------------------------------------------------------------------

#takes 1000 github issues from a chosen dataset and encodes them using title and body
def get_embeddings():
    #load basic github dataset
    dataset = load_dataset("sharjeelyunus/github-issues-dataset", split="train[:1000]")

    #extract title and first 500 characters of body for encoding
    texts_to_encode = []
    for sample in dataset:
        title = sample.get("title")
        body = (sample.get("body") or "")[:500]
        text = f"{title}\n{body}".strip()
        texts_to_encode.append(text)

    #let's get them embeddings
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts_to_encode, show_progress_bar=True)
    return embeddings

#convert embeddings to 2D umap and scale to [0,1] (using 2D umap for visuals and isn't directly used for learning)
def to_umap_2d(embeddings)
    reducer = umap.UMAP(n_components=2, random_state=42)
    scaler = MinMaxScaler()
    coords_2d = reducer.fit_transform(embeddings)
    coords_norm = scaler.fit_transform(coords_2d)
    return coords_norm

#inserts the data into sample table
def insert_samples():
    pass
    

def main():
    harvest_samples()
    init_db()
    reset_db()


if __name__ == "__main__":
    main()
