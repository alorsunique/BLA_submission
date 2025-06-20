import ollama
import pandas as pd
import os
from pathlib import Path
import json
import numpy as np

from ollama import chat
from ollama import ChatResponse


def load_embed_json(json_embed_path):

    embedding_list = []

    if json_embed_path.exists():

        with open(json_embed_path, 'r') as embed_json:
            embedding_list = json.load(embed_json)

    else:
        print("Not Found")

    return embedding_list
    

def cosine_similarity_sort(query_vector, embedding_list):

    embed_cos_list = []

    count = 0

    for embedding_vector in embedding_list:

        a = np.array(query_vector)
        b = np.array(embedding_vector)

        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        cos_similar_value = dot_product / (norm_a*norm_b)
        
        embed_cos_tuple = (cos_similar_value,count)

        embed_cos_list.append(embed_cos_tuple)

        count += 1

    
    sorted_embed_cos_list = sorted(embed_cos_list, key=lambda x: x[0], reverse=True)

    return sorted_embed_cos_list



if __name__ == "__main__":

    df = pd.read_csv("dataset.csv") 


    print("Hey")

    json_path = Path("embedding.json")

    embedding_list = load_embed_json(json_path)

    print(embedding_list)


    user_input = input("Input: ")

    user_input_embedding_values = ollama.embeddings(model='nomic-embed-text:v1.5', prompt=user_input)['embedding']

    # print(type(user_input_embedding_values))

    embed_cos_list = cosine_similarity_sort(user_input_embedding_values, embedding_list)

    # print(embed_cos_list)

    count = 0

    max_count = 3

    print(f"Top {max_count} results based on similarity to query")

    top_list = []

    while count < max_count:

        current_row = df.iloc[embed_cos_list[count][1]]

        cos_score = embed_cos_list[count][0]

        current_name = current_row['name']
        current_category = current_row['category']
        current_location = current_row['location']
        current_description = current_row['description']

        print(f"Name: {current_name} | Category: {current_category} | Location: {current_location} | Description: {current_description} | Score: {cos_score}")

        if cos_score > 0.75:
            top_list = (current_name, current_location, current_category, current_description)

        count += 1

    if len(top_list) == 0:
        print(f"Did not find relevant match")
    else:
        message_list = []

        system_prompt = '''
            You are a Business Lookup Assistant. You will help the user look for business that closely aligns with their requests.
        '''

        # Appends the system prompt
        message_list.append(
            {
                "role": "system",
                "content": system_prompt,
            }
        )

        message_list.append(
            {
                "role": "user",
                "content": f"This is the user query: {user_input}",
            }
        )

        message_list.append(
            {
                "role": "user",
                "content": f"This is the most relevant result: {top_list[0]}",
            }
        )

        message_list.append(
            {
                "role": "user",
                "content": f"Synthesize a response to answer the query based on the most relevant result. Make it engaging. Tell the name of the shop, the location, the type and the description. Do not offer any other help or extra information.",
            }
        )

        response: ChatResponse = chat(
            model='gemma3:1b-it-qat',
            messages=message_list,
        )

        print(f"\n\nLLM Response")

        response_message = response.message.content

        print(response_message)

        