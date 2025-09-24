## Retrieval/ranker for player discovery

import os
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import pandas as pd
from dotenv import load_dotenv

# load env vars 
load_dotenv()
data = pd.read_csv("data/bets.csv")

# config 
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim
INDEX_NAME = "bets-semantic-search"
API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE = Pinecone(api_key=API_KEY)

# index setup 
def create_idx():
    indices = PINECONE.list_indexes()
    lookup = {i["name"]: i for i in indices}
    if INDEX_NAME not in lookup:
        index = PINECONE.create_index(
            name=INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    else: 
        index = lookup[INDEX_NAME]
    return index

create_idx()


# ----- Embeddings -----
# model = SentenceTransformer(EMBED_MODEL_NAME)
# embeddings = model.encode(data["text"].tolist(), normalize_embeddings=True)
# embeddings = np.asarray(embeddings)
# dim = embeddings.shape[1]


# ----- Pinecone -----





# def stable_id(record: Dict[str, Any]) -> str:
#     """Deterministic ID so re-ingests upsert same vector."""
#     return hashlib.sha1(record.encode("utf-8")).hexdigest()

# def to_metadata(row: pd.Series) -> Dict[str, Any]:
#     md = {
#         "sport": row["o_category_name"],
#         "user": row["mb_account_id"],
#         "date": row["ml_placed_date_time"],
#         "search_text": row["text"],
#         "raw": row["raw"],
#         "model": EMBED_MODEL_NAME,
#         "embedding_norm": "cosine"
#     }
#     return {k: v for k, v in md.items() if v is not None}


# vectors = []
# for i, row in data["text"].iterrows():
#     vid = stable_id(row)
#     vectors.append({"id": vid, "values": embeddings[i].tolist(), "metadata": to_metadata(row)})

# # Batch upserts (Pinecone can take fairly large batches; 100–200 is safe)
# BATCH = 100
# for start in range(0, len(vectors), BATCH):
#     index.upsert(vectors=vectors[start:start+BATCH])


# # def embed_texts(texts: List[str]) -> List[List[float]]:
# #     return model.encode(texts, normalize_embeddings=True).tolist()

# # def to_vectors(df: pd.DataFrame, id_cols: List[str] = ID_COLS) -> List[Dict[str, Any]]:
# #     vectors = []
# #     for _, row in df.iterrows():
# #         text = row_to_search_text(row)
# #         # choose a stable vector id (fallback to index if missing)
# #         vec_id = as_str(row.get("ml_mult_leg_id")) or as_str(row.get("ml_mult_id")) or as_str(row.get("o_feed_selection_id"))
# #         if not vec_id:
# #             vec_id = f"row-{_}"
# #         vectors.append({"id": vec_id, "text": text, "metadata": extract_metadata(row)})
# #     return vectors