#!/usr/bin/env python
import argparse
import asyncio
from typing import List

from pydantic import BaseModel, TypeAdapter
from pydantic_ai import Agent, UrlContextTool, WebSearchTool, PromptedOutput
from pydantic_ai.models.gemini import GeminiModelSettings 
from dotenv import load_dotenv
import data_model
import tools

def get_model_settings(temperature: float = 0.25) -> GeminiModelSettings:
    return GeminiModelSettings(
        temperature=0.25,
        gemini_safety_settings=[
            {
                'category': 'HARM_CATEGORY_DANGEROUS_CONTENT',
                'threshold': 'BLOCK_ONLY_HIGH',
            }
        ]
    )

def create_agent() -> Agent:
    return Agent(
        "gemini-2.5-flash",
        model_settings=get_model_settings(),
        retries=2,
        system_prompt=f"""
        You are a research analyst. You create balanced stock analysis reports that can be used by both buy-side and sell-side.
        """,
    )

async def generate_company(ticker) -> data_model.Company:
    agent = create_agent()
    
    @agent.tool_plain
    async def get_stock_price(ticker: str) -> float:
        """Get the latest closing stock price of a publicly traded stock."""
        return await tools.get_stock_price_mock(ticker)
    
    print("Calling Gemini via Pydantic for Company Info and using Yahoo Finance for stock price")
    result = await agent.run(
        f"""
        Get company information for {ticker}. If the company is commonly known by another name, provide that name also.
        For example, for WW, the name would be "WW International Inc., formerly Weight Watchers International, Inc."
        </example>
        """,
        output_type=data_model.Company
    )
    return result.output

async def generate_company_overview(ticker) -> data_model.CompanyOverview:
    print("Calling Gemini via Pydantic for Company Overview") 
    agent_without_tools = create_agent()
    result = await agent_without_tools.run(
        f"""Provide a company overview of {ticker}: what industry it is in, what the company does, and its key products or segments.
        """,
        output_type=data_model.CompanyOverview
    )
    return result.output


async def generate_key_financials(ticker: str) -> data_model.KeyFinancialData:
    print("Using Gemini's UrlContext for Key Financial Data")
    agent_with_url = Agent(
        "gemini-2.5-flash",
        model_settings=get_model_settings(temperature=0.1),
        retries=2,
        builtin_tools=[UrlContextTool()]
    )
    result = await agent_with_url.run(
        f"""
        Get key financial data on {ticker} from https://finance.yahoo.com/quote/{ticker}/financials/
        Do not make up any numbers.
        """,
        output_type=PromptedOutput(data_model.KeyFinancialData) # Gemini does not support output tools and built-in tools at the same time. Use `output_type=PromptedOutput(...)
    )
    return result.output


async def generate_case(buy_side: bool, ticker: str, n_points: int = 3) -> List[data_model.BulletPoint]:
    which_side = "buy-side" if buy_side else "sell-side"
    print(f"""Using Gemini's WebSearch tool for {which_side} points""")
    agent_with_search = Agent(
        "gemini-2.5-flash",
        model_settings=get_model_settings(temperature=0.1),
        retries=2,
        system_prompt=f"""
        Search the web and find relatively recent analyst reports and summarize
        the key points to make the {which_side} case for the given stock.
        """,
        builtin_tools=[WebSearchTool()]
    )
    result = await agent_with_search.run(
        f"""
        Make the {which_side} case for {ticker}. Produce exactly {n_points} points.
        Each title should be less than 10 words, and each point should be less than 200 words.
        """,
        output_type=PromptedOutput(List[data_model.BulletPoint])
    )

    # Reflection (once)
    def validate(points: List[data_model.BulletPoint]) -> List[str]:
        error_messages = []
        if len(points) != n_points:
            error_messages.append(f"You wrote {len(points)} points, not {n_points}")
        for bulletno, bullet in enumerate(points):
            num_title_words = len(bullet.title.split())
            if num_title_words > 10:
                error_messages.append(f"The title of bullet no {bulletno+1} is too long")
            num_bullet_words = len(bullet.explanation.split())
            if num_bullet_words > 200:
                error_messages.append(f"The explanation in bullet no {bulletno+1} is too long")
        return error_messages
    
    error_messages = validate(result.output)
    if len(error_messages) == 0:
        return result.output
    
    # second attempt
    print(f"Reflection: invoking model a second time to fix {len(error_messages)} errors")
    prev_result = TypeAdapter(List[data_model.BulletPoint]).dump_json(result.output)
    errors = '\n'.join(error_messages)
    result2 = await agent_with_search.run(
        f"""
        Make the {which_side} case for {ticker}. Produce exactly {n_points} points.
        Each title should be less than 10 words, and each point should be less than 200 words.

        Last time, you produced {prev_result}, which had the following problems:
        {errors}
        Please fix.
        """,
        output_type=List[data_model.BulletPoint]
    )
    error_messages2 = validate(result2.output)

    # return the better one
    if len(error_messages) < len(error_messages2):
        return result.output
    else:
        return result2.output


async def generate_analysis(ticker) -> data_model.StockReport:
    """Generates a stock analysis using the Gemini API."""
    data = await asyncio.gather(
        generate_company(ticker),
        generate_company_overview(ticker),
        generate_key_financials(ticker),
        generate_case(True, ticker, 3),
        generate_case(False, ticker, 3)
    )
    data_as_dict = {
        "company": data[0],
        "company_overview": data[1],
        "key_financial_data": data[2],
        "buy_case": data[3],
        "sell_case": data[4]
    }
    return data_model.StockReport(**data_as_dict)


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
