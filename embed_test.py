import ollama
import pandas as pd
import os
from pathlib import Path
import json

def line_chunk(data_df):
    line_chunk_list = []
    for index, row in data_df.iterrows():
        chunk = f"{row['name']} is a {row['category']} in {row['location']}. This is the description of the {row['category']}: {row['description']}"
        print(chunk)

        line_chunk_list.append(chunk)

    return line_chunk_list


def get_embed(line_chunk_list):
    embedding_list = []
    for chunk in line_chunk_list:
        embedding_values = ollama.embeddings(model='nomic-embed-text:v1.5', prompt=chunk)['embedding']

        print(len(embedding_values))
        embedding_list.append(embedding_values)

    return(embedding_list)

def store_embed_json(embedding_list, json_embed_path):
    

    if json_embed_path.exists():
        os.remove(json_embed_path)
    
    with open(json_embed_path, 'w') as embed_json:
        json.dump(embedding_list,embed_json)

if __name__ == "__main__":

    df = pd.read_csv("dataset.csv") 

    print(df)

    line_chunk_list = line_chunk(df)

    json_path = Path("embedding.json")

    embedding_list = get_embed(line_chunk_list)

    store_embed_json(embedding_list, json_path)



#embed = ollama.embeddings(model='nomic-embed-text:v1.5', prompt='The house is called')

#print(type(embed))

# print(embed)
