import os 
from dotenv import load_dotenv
import pandas as pd 
from sentence_transformers import SentenceTransformer
load_dotenv()

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim
SENTENCE_TRANSFORMER = SentenceTransformer(EMBED_MODEL_NAME)

INDEX_NAME = "bets-semantic-search"
API_KEY = os.environ["PINECONE_API_KEY"]

DATA = pd.read_csv("data/bets.csv")
DATA['SearchText'] = DATA['RawBetInformation'] + " | " + DATA['FixtureName'] + " | " + DATA['OutcomeDescription']
