import requests
import os
import json
import pandas as pd
import numpy as np
import joblib
import sys
sys.stdout.reconfigure(encoding='utf-8')
from sklearn.metrics.pairwise import cosine_similarity


def create_embedding(text_list):
    # https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })

    print("status:", r.status_code)
    print("Response:", r.text)

    if r.status_code != 200:
        raise Exception("Ollama API request failed")
    
    embedding = r.json()["embeddings"] 
    return embedding


jsons = os.listdir("newjsons")  # List all the jsons 
my_dicts = []
chunk_id = 0

for json_file in jsons:
    with open(f"newjsons/{json_file}", "r", encoding="utf-8") as f:
        content = json.load(f)
    print(f"Creating Embeddings for {json_file}")
    embeddings = create_embedding([c['text'] for c in content['chunks']])
       
    for i, chunk in enumerate(content['chunks']):
        chunk['chunk_id'] = chunk_id
        chunk['embedding'] = embeddings[i]
        chunk_id += 1
        my_dicts.append(chunk) 
    
# print(my_dicts)

df = pd.DataFrame.from_records(my_dicts)
joblib.dump(df, 'embeddings.joblib')
#print(df)
# incoming_query = input("Ask me a question: ")
# question_embedding = create_embedding([incoming_query])[0] 
# #print(question_embedding)
# # a = create_embedding(["Cat sat on the mat", "Harry dances on a mat"])
# # print(a)

# # print(np.vstack(df['embedding'].values))
# # print(np.vstack(df['embedding']).shape)

# similarities = cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten()
# print(similarities)
# top_results = 3
# max_indx = similarities.argsort()[::-1][0:top_results]
# print(max_indx)
# new_df = df.loc[max_indx] 
# print(new_df[["title", "number", "text"]])