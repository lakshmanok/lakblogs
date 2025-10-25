#!/usr/bin/env python
import argparse
import asyncio
from typing import List

from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModelSettings 
from dotenv import load_dotenv
from data_model import StockReport
import tools

async def generate_analysis(ticker) -> StockReport:
    """Generates a stock analysis using the Gemini API."""
    price = await tools.get_stock_price_mock(ticker)

    print("Calling Gemini for StockReport")
    model_settings = GeminiModelSettings(
                temperature=0.25,
                gemini_safety_settings=[
                    {
                        'category': 'HARM_CATEGORY_DANGEROUS_CONTENT',
                        'threshold': 'BLOCK_ONLY_HIGH',
                    }
                ]
            )
    agent = Agent(
        "gemini-2.5-flash",
        output_type=StockReport,
        model_settings=model_settings,
        retries=2,
        system_prompt=f"""
        You are research analyst. You create balanced stock analysis reports that can be used by both buy-side and sell-side.
        Return structured JSON in the desired format.
        """,
    )
    @agent.tool_plain
    async def get_stock_price(ticker: str) -> float:
        """Get the latest closing stock price of a publicly traded stock."""
        return await tools.get_stock_price_mock(ticker)

    result = await agent.run(
        f"Create a report on {ticker}."
    )
    print("data received")
    obj: StockReport = result.output
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
