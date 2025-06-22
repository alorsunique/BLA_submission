# This script should take in the top results and generate a response for the query

import embed_retrieve
import pandas as pd
from pathlib import Path
from ollama import chat
from ollama import ChatResponse
from datetime import datetime
import random
import json


# For logging booking
def get_formatted_time():
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    random_id = f"{random.randint(0, 99999999):08d}"
    return f"{timestamp}_{random_id}"


# Generate the response
def generate_response(top_list, user_input, model_name):

    message_list = []

    # System prompt
    system_prompt = """
        You are a Business Lookup Assistant. You will help the user look for business that closely aligns with their requests.
    """

    # Appends the system prompt
    message_list.append(
        {
            "role": "system",
            "content": system_prompt,
        }
    )

    # Appends the user query
    message_list.append(
        {
            "role": "user",
            "content": f"This is the user query: {user_input}.",
        }
    )

    # If no match
    if len(top_list) == 0:

        message_list.append(
            {
                "role": "user",
                "content": f"After searching our database, there is no relevant result.",
            }
        )

        message_list.append(
            {
                "role": "user",
                "content": f"Create a response informing the user that we did not find a good match in our database. Do not offer any other help or extra information.",
            }
        )

        message_list.append(
            {
                "role": "user",
                "content": f"Just reply with the response of having no match.",
            }
        )

        message_list.append(
            {
                "role": "user",
                "content": f"Also keep it simple.",
            }
        )

        shop_tuple = ("No Match", "No Match", "No Match")

        result_found = False

    # If match
    else:

        message_list.append(
            {
                "role": "user",
                "content": f"This is the most relevant result:",
            }
        )

        name = top_list[0][0]
        category = top_list[0][1]
        location = top_list[0][2]
        description = top_list[0][3]
        menu = top_list[0][4]

        message_list.append(
            {
                "role": "user",
                "content": f"Name: {name}\nCategory: {category}\nLocation: {location}\nDescription: {description}\nMenu: {menu}\n",
            }
        )

        message_list.append(
            {
                "role": "user",
                "content": f"Synthesize a response to answer the query based on the most relevant result. Make it engaging.",
            }
        )

        message_list.append(
            {
                "role": "user",
                "content": f"Make sure to tell the name of the shop, the location, the type, the description, and the menu. Do not offer any other help or extra information.",
            }
        )

        message_list.append(
            {
                "role": "user",
                "content": f"Also keep it simple.",
            }
        )

        shop_tuple = (name, category, location)

        result_found = True

    # Uses the specified model to generate response
    response: ChatResponse = chat(
        model=model_name,
        messages=message_list,
    )

    response_message = response.message.content

    return response_message, result_found, shop_tuple


# Parse the booking request of the user
def book_table_command(user_input, shop_tuple, model_name):

    message_list = []

    # System prompt
    system_prompt = f"""
        You are a booking table assistant. You will check if the user expressed their desire to book a table at {shop_tuple[0]}.
    """

    # Appends the system prompt
    message_list.append(
        {
            "role": "system",
            "content": system_prompt,
        }
    )

    message_list.append(
        {
            "role": "system",
            "content": f"Note that they have been asked already if they want to book a table at {shop_tuple[0]}.",
        }
    )

    message_list.append(
        {
            "role": "system",
            "content": f"This is their response.",
        }
    )

    message_list.append(
        {
            "role": "user",
            "content": user_input,
        }
    )

    message_list.append(
        {
            "role": "system",
            "content": f"If the user input is gibberish or not related to booking a table, NO must be the response.",
        }
    )

    message_list.append(
        {
            "role": "system",
            "content": f"Respond with either YES or NO only. No punctuations in the response. Either YES or NO only. I repeat, YES or NO only.",
        }
    )

    response: ChatResponse = chat(
        model=model_name,
        messages=message_list,
    )

    response_message = response.message.content
    print(response_message)

    if response_message.lower().strip() == "yes":

        user_id = get_formatted_time()
        shop_name = shop_tuple[0]
        shop_category = shop_tuple[1]
        shop_location = shop_tuple[2]
        user_book_message = user_input

        booking_entry_dict = {
            "user_id": user_id,
            "shop_name": shop_name,
            "shop_category": shop_category,
            "shop_location": shop_location,
            "message": user_book_message,
        }

        book_status = True

    else:
        booking_entry_dict = {}

        book_status = False

    return book_status, booking_entry_dict


# This modify the JSON containing the booking list
def modify_book_json(json_book_path, booking_entry_dict):

    # Create the JSON file
    if not json_book_path.exists():
        with open(json_book_path, "w") as file:
            json.dump([], file)

    with open(json_book_path, "r") as file:
        try:
            booking_data = json.load(file)
        except json.JSONDecodeError:
            booking_data = []

    booking_data.append(booking_entry_dict)

    # Write new data
    with open(json_book_path, "w") as file:
        json.dump(booking_data, file, indent=4)


# Main
if __name__ == "__main__":

    df = pd.read_csv("expanded_dataset.csv")
    json_path = Path("embedding.json")
    llm_name = "gemma3:1b-it-qat"

    json_book_path = Path("booking.json")

    user_input = input("Input: ")

    embedding_list = embed_retrieve.load_embed_json(json_path)

    top_list = embed_retrieve.get_top_list(df, user_input, embedding_list)

    response_message, result_found, shop_tuple = generate_response(
        top_list, user_input, llm_name
    )

    print(response_message)

    if result_found:
        print("Do you want to book a table?")

        user_input = input("Input: ")

        book_status, booking_entry_dict = book_table_command(
            user_input, shop_tuple, llm_name
        )

        if book_status:
            modify_book_json(json_book_path, booking_entry_dict)
