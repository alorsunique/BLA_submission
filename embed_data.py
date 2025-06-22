# This script should embed the dataset

import ollama
import pandas as pd
import os
from pathlib import Path
import json


# Chunks the data
def line_chunk(data_df):
    line_chunk_list = []
    for index, row in data_df.iterrows():
        name = row["name"]
        category = row["category"]
        location = row["location"]
        description = row["description"]
        menu = row["menu"]

        chunk = f"Name: {name}\nCategory: {category}\nLocation: {location}\nDescription: {description}\nMenu: {menu}"
        print(chunk)

        line_chunk_list.append(chunk)

    return line_chunk_list


# Get the embedding
def get_embed(line_chunk_list, embed_model):
    embedding_list = []
    for chunk in line_chunk_list:
        embed_response = ollama.embeddings(model=embed_model, prompt=chunk)
        embedding_values = embed_response["embedding"]
        embedding_list.append(embedding_values)

    return embedding_list


# Store the embeddingg in a JSON file
def store_embed_json(embedding_list, json_embed_path):
    if json_embed_path.exists():
        os.remove(json_embed_path)

    with open(json_embed_path, "w") as embed_json:
        json.dump(embedding_list, embed_json)


# Main
if __name__ == "__main__":
    # Load the dataset
    df = pd.read_csv("expanded_dataset.csv")
    embed_model = "nomic-embed-text:v1.5"

    line_chunk_list = line_chunk(df)
    json_path = Path("embedding.json")
    embedding_list = get_embed(line_chunk_list, embed_model)
    store_embed_json(embedding_list, json_path)

    print(f"Stored: {len(embedding_list)}")
