from pinecone import Pinecone
from src.constants import API_KEY
from src.constants import INDEX_NAME, SENTENCE_TRANSFORMER, DATA
from pinecone import ServerlessSpec
import numpy as np

PINECONE = Pinecone(api_key=API_KEY)

def create_idx(index_name=INDEX_NAME):
    indices = PINECONE.list_indexes()
    lookup = {i["name"]: i for i in indices}
    if INDEX_NAME not in lookup:
        index = PINECONE.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    else: 
        index = lookup[INDEX_NAME]
    return index

def create_embeddings(model=SENTENCE_TRANSFORMER):
    embeddings = model.encode(DATA["SearchText"].tolist(), normalize_embeddings=True)
    embeddings = np.asarray(embeddings)
    ## TODO: SEND THESE GUYS TO PINECONE INDEX THAT WE DEFINED UP
    return embeddings