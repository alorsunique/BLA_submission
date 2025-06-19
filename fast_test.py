from fastapi import FastAPI
from pydantic import BaseModel

# Create a FastAPI instance
app = FastAPI()

# Define the input model using Pydantic
class UserInput(BaseModel):
    name: str
    age: int

# Create a POST endpoint that takes user input
@app.post("/submit/")
async def receive_input(user_input: UserInput):
    return {
        "message": f"Hello {user_input.name}, you are {user_input.age} years old!"
    }