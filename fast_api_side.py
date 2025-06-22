# Set up the API
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi import Request
from contextlib import asynccontextmanager
import pandas as pd
from pathlib import Path

import embed_retrieve
import augment_generate


# Load the dataset and embeddings at startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        df = pd.read_csv("expanded_dataset.csv")
        json_path = Path("embedding.json")
        embedding_list = embed_retrieve.load_embed_json(json_path)

        # Store in app state
        app.state.dataset = df
        app.state.embeddings = embedding_list

        print("Data loaded successfully.")
    except Exception as e:
        print(f"Error loading data: {e}")

    yield


# Create the FastAPI app
app = FastAPI(lifespan=lifespan)


class QueryInput(BaseModel):
    query: str


# Endpoint that handles user queries
@app.post("/submit/")
async def submit_query(input_data: QueryInput, request: Request):
    df = request.app.state.dataset
    embedding_list = request.app.state.embeddings
    llm_name = "gemma3:1b-it-qat"

    user_input = input_data.query

    top_list = embed_retrieve.get_top_list(df, user_input, embedding_list)
    response_message, result_found, shop_tuple = augment_generate.generate_response(
        top_list, user_input, llm_name
    )

    return {
        "response": response_message,
        "found": result_found,
        "shop_tuple": shop_tuple,
    }


class BookingInput(BaseModel):
    user_input: str
    shop_tuple: tuple

# Endpoint that handles booking
@app.post("/book/")
async def book_table(input_data: BookingInput):
    from pathlib import Path

    json_book_path = Path("booking.json")  # or whatever your booking file is
    llm_name = "gemma3:1b-it-qat"

    book_status, booking_entry_dict = augment_generate.book_table_command(
        input_data.user_input, input_data.shop_tuple, llm_name
    )

    if book_status:
        augment_generate.modify_book_json(json_book_path, booking_entry_dict)

    return {"booked": book_status, "entry": booking_entry_dict if book_status else None}
