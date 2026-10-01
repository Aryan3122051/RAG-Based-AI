import pandas as pd 
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 
import joblib 
import requests

def create_embedding(text_list):
    # https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })

    print("status:", r.status_code)
    #print("Response:", r.text)

    if r.status_code != 200:
        raise Exception("Ollama API request failed")
    
    embedding = r.json()["embeddings"] 
    return embedding

def inference(prompt):
    r = requests.post("http://localhost:11434/api/generate", json={
        # "model": "deepseek-r1",
         "model": "llama3.2",
        "prompt": prompt,
        "stream": False,
    })
    
    response = r.json()
    print(response)
    return response

df = joblib.load('embeddings.joblib')


incoming_query = input("Ask me a question: ")
question_embedding = create_embedding([incoming_query])[0] 
#print(question_embedding)
# a = create_embedding(["Cat sat on the mat", "Harry dances on a mat"])
# print(a)

# print(np.vstack(df['embedding'].values))
# print(np.vstack(df['embedding']).shape)

similarities = cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten()
#print(similarities)
top_results = 3
max_indx = similarities.argsort()[::-1][0:top_results]
#print(max_indx)
new_df = df.loc[max_indx] 
#print(new_df[["title", "number", "text"]])

prompt = f''' I am teaching web development in my sigma web development course. Here are video subtitles chunks containing video title, video number, start time in second, end time in seconds, the text at that time:

{new_df[["title", "number", "text", "start", "end"]].to_json(orient="records")}
-----------------------------
"{incoming_query}"
user asked this question related to the video chunks, you have to answer in a human way(dont mention the above format, its just or you) where and how much content is taught in one video (in which video and at what timestamp) and guide the user to go to that particular video. If user asks unrelated questions, tell him that you can only answer question related to the course

'''

with open("prompt.txt", "w") as f:
    f.write(prompt)

response = inference(prompt)["response"]
print(response)

with open("response.txt", "w") as f:
    f.write(response)

# for index, item in newdf.iterrows():
#     print(index, item["title"], item["number"], item["text"], item["start"], item["end"])