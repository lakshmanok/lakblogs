import pydantic_ai
from pydantic_ai.models.gemini import GeminiModel
from dotenv import load_dotenv
import os

load_dotenv("../keys.env")

def default_model() -> pydantic_ai.models.Model:
    model = GeminiModel('gemini-1.5-flash', api_key=os.getenv('GOOGLE_API_KEY'))
    return model

def agent() -> pydantic_ai.Agent:
    return pydantic_ai.Agent(default_model())