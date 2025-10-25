#!/usr/bin/env python
import argparse
import asyncio
from typing import List

from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents.middleware import wrap_tool_call
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import ToolMessage
from langchain.tools import tool
from dotenv import load_dotenv
from data_model import StockReport
import tools

@tool
def get_stock_price(ticker: str) -> float:
    """Get the latest closing stock price of a publicly traded stock."""
    return asyncio.run(
        # no way to call async tools!
        tools.get_stock_price_mock(ticker)
    )

@wrap_tool_call
def cache_tool_results(request, handler):
    """Cache results of stock & date and return results from cache if possible."""
    try:
        return handler(request)
    except Exception as e:
        # Return a custom error message to the model
        return ToolMessage(
            content=f"Tool error: Please check your input and try again. ({str(e)})",
            tool_call_id=request.tool_call["id"]
        )


async def generate_analysis(ticker) -> StockReport:
    """Generates a stock analysis using the Gemini API."""
    price = await tools.get_stock_price_mock(ticker)
    prompt = f"Create a stock analysis report for {ticker}."

    print("Calling Gemini for StockReport")
    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.25
    )
    agent = create_agent(
        model,
        system_prompt=f"""
        You are research analyst. You create balanced stock analysis reports that can be used by both buy-side and sell-side.
        Return structured JSON in the desired format.
        """,
        tools=[get_stock_price],
        # Bug: with ProviderStrategy https://github.com/langchain-ai/langchainjs/issues/8585
        response_format=ToolStrategy(StockReport)
    )
    result = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": f"Create a report on {ticker}."}
            ]
        }
    )
    print("data received")
    obj: StockReport = result["structured_response"]
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
