# This script should retrieve the top entries

import ollama
import pandas as pd
from pathlib import Path
import json
import numpy as np


# Load the embedding list from JSON
def load_embed_json(json_embed_path):
    embedding_list = []

    if json_embed_path.exists():
        with open(json_embed_path, "r") as embed_json:
            embedding_list = json.load(embed_json)

    else:
        print("Not Found")

    return embedding_list


# Get the cosine similarity and sort the values
def cosine_similarity_sort(query_vector, embedding_list):

    embed_cos_list = []

    count = 0

    for embedding_vector in embedding_list:

        a = np.array(query_vector)
        b = np.array(embedding_vector)

        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        cos_similar_value = dot_product / (norm_a * norm_b)

        embed_cos_tuple = (cos_similar_value, count)

        embed_cos_list.append(embed_cos_tuple)

        count += 1

    # Sort to get the best result at the top
    sorted_embed_cos_list = sorted(embed_cos_list, key=lambda x: x[0], reverse=True)

    return sorted_embed_cos_list


# Get the top 3 matches, if available
def get_top_list(dataset_df, user_input, embedding_list):
    # Embed user input
    user_input_embedding_values = ollama.embeddings(
        model="nomic-embed-text:v1.5", prompt=user_input
    )["embedding"]

    embed_cos_list = cosine_similarity_sort(user_input_embedding_values, embedding_list)

    count = 0
    max_count = 3

    print(f"Top {max_count} results based on similarity to query")

    top_list = []

    while count < max_count:

        current_row = dataset_df.iloc[embed_cos_list[count][1]]

        cos_score = embed_cos_list[count][0]

        name = current_row["name"]
        category = current_row["category"]
        location = current_row["location"]
        description = current_row["description"]
        menu = current_row["menu"]

        print(
            f"Name: {name}\nCategory: {category}\nLocation: {location}\nDescription: {description}\nMenu: {menu}\nScore: {cos_score}\n"
        )

        count += 1

        if cos_score > 0.5:
            top_list.append((name, category, location, description, menu))

    return top_list


# Main
if __name__ == "__main__":
    df = pd.read_csv("expanded_dataset.csv")
    json_path = Path("embedding.json")

    embedding_list = load_embed_json(json_path)

    user_input = input("Input: ")

    top_list = get_top_list(df, user_input, embedding_list)

    print(top_list)
