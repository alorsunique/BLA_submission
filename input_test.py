from fastapi import FastAPI
from pydantic import BaseModel

from ollama import chat
from ollama import ChatResponse


# Create a FastAPI instance
app = FastAPI()

# Define the input model using Pydantic
class UserInput(BaseModel):
    input_string: str

# Create a POST endpoint that takes user input
@app.post("/submit/")
async def receive_input(user_input: UserInput):

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
            "content": user_input.input_string ,
        }
    )

    user_prompt = "That is the request made by the user"

    message_list.append(
        {
            "role": "user",
            "content": user_prompt,
        }
    )

    response: ChatResponse = chat(
        model='gemma3:1b-it-qat',
        messages=message_list,
    )


    response_message = response.message.content

    return {
        "message": f"{response_message}"
    }