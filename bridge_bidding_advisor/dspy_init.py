import os
import dotenv
import dspy


def init_gemini_pro(temperature: float = 0.0):
    """
    Initializes dspy to use Gemini as the language model.
    """
    dotenv.load_dotenv("api_keys.env")
    api_key = os.getenv("GOOGLE_API_KEY")
    gemini = dspy.Google("models/gemini-1.0-pro",
                         api_key=api_key,
                         temperature=temperature)
    dspy.settings.configure(lm=gemini)


def init_gpt35(temperature: float = 0.0):
    """
    Initializes dspy to use OpenAI GPT 3.5 as the language model.
    """
    dotenv.load_dotenv("api_keys.env")
    api_key = os.getenv("OPENAI_API_KEY")
    gpt35 = dspy.OpenAI(model="gpt-3.5-turbo",
                        api_key=api_key,
                        temperature=temperature)
    dspy.settings.configure(lm=gpt35)
