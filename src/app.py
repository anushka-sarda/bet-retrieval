from fastapi import FastAPI
import src.search as search

app = FastAPI() 

@app.get("/search/{query}")
def results(query: str):
    return search.get_results(query)