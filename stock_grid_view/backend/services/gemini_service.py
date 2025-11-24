import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

# Configure API key
# Ensure GOOGLE_API_KEY is set in environment variables
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key or api_key == "your_api_key_here":
    print("Warning: GOOGLE_API_KEY is not set in .env")

genai.configure(api_key=api_key)

model = genai.GenerativeModel('gemini-2.5-flash')

def generate_cell_content(ticker: str, column_description: str) -> str:
    if not api_key or api_key == "your_api_key_here":
        return "Set API Key"
        
    prompt = f"For the stock ticker '{ticker}', provide a short, concise value for the column described as '{column_description}'. Return ONLY the value, no markdown, no explanations."
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error generating content for {ticker} - {column_description}: {e}")
        return "Error"

def resolve_ticker(input_text: str) -> str:
    if not api_key or api_key == "your_api_key_here":
        return input_text.upper()
        
    prompt = f"What is the stock ticker symbol for '{input_text}'? Return ONLY the ticker symbol (e.g. for 'Apple' return 'AAPL'). If it's already a ticker or you are unsure, return the input uppercase."
    try:
        response = model.generate_content(prompt)
        return response.text.strip().upper()
    except Exception as e:
        print(f"Error resolving ticker for {input_text}: {e}")
        return input_text.upper()
