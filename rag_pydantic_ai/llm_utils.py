import pydantic_ai
from typing import Final, LiteralString
from pydantic_ai.models.gemini import GeminiModel
from dotenv import load_dotenv
import os

load_dotenv("../keys.env")

def agent() -> pydantic_ai.Agent:
    model = GeminiModel('gemini-1.5-flash', api_key=os.getenv('GOOGLE_API_KEY'))
    return pydantic_ai.Agent(model)

