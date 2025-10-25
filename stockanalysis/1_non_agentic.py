#!/usr/bin/env python
import argparse
import asyncio
from typing import List

from google import genai
from dotenv import load_dotenv
from data_model import StockReport
import tools

async def generate_analysis(ticker) -> StockReport:
    """Generates a stock analysis using the Gemini API."""
    price = await tools.get_stock_price_mock(ticker)
    prompt = f"Create a stock analysis report for {ticker}. The current price is ${price:.2f}."

    print("Calling Gemini for StockReport")
    client = genai.Client()
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={
            "response_mime_type" : "application/json",
            "response_schema" : StockReport,
        }
    )
    print("data received")
    json_data = response.text
    print(json_data)
    obj: StockReport = response.parsed
    return obj


async def main():
    """The main function."""
    parser = argparse.ArgumentParser(description="Create a stock analysis report.")
    parser.add_argument("ticker", help="The stock ticker symbol.")
    args = parser.parse_args()

    load_dotenv(dotenv_path="../keys.env") # GEMINI_API_KEY

    report = await generate_analysis(args.ticker)
    print(tools.render_report(report))


if __name__ == "__main__":
    asyncio.run(main())
