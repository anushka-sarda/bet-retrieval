## Retrieval/ranker for player discovery
from src.constants import SENTENCE_TRANSFORMER

def get_results(query, top_k=5):
    query_embedding = SENTENCE_TRANSFORMER.encode([query])
    # similarities = cosine_similarity(query_embedding, embeddings)[0]
    # TODO: get results from PINECONE
    return []

