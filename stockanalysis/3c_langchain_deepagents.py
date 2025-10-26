#!/usr/bin/env python
import argparse
import asyncio
from pydantic import BaseModel
from typing import List

from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory
# from langchain.agents.middleware import wrap_tool_call -- this is cool, allowing you to intercept anywhere
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import ToolMessage
from langchain.tools import tool
from google.ai.generativelanguage_v1beta.types import Tool as GenAITool
from dotenv import load_dotenv
import data_model
import tools
from urllib.request import urlopen
from deepagents import create_deep_agent

@tool
def get_stock_price(ticker: str) -> float:
    """Get the latest closing stock price of a publicly traded stock."""
    return asyncio.run(
        # no way to call async tools!
        tools.get_stock_price_mock(ticker)
    )

def create_model(temperature: float = 0.25) -> ChatGoogleGenerativeAI:
    gemini_safety_settings = {}
    gemini_safety_settings[HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT] = HarmBlockThreshold.BLOCK_ONLY_HIGH
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=temperature,
        safety_settings=gemini_safety_settings
    )


async def generate_company(ticker: str, additional_instructions: str = "") -> data_model.Company:
    """Generate basic company information"""
    model = create_model()
    agent = create_agent(
        model,
        system_prompt=f"""
        You are research analyst. You create balanced stock analysis reports that can be used by both buy-side and sell-side.
        Return structured JSON in the desired format.
        """,
        tools=[get_stock_price],
        # Bug: with ProviderStrategy https://github.com/langchain-ai/langchainjs/issues/8585
        response_format=ToolStrategy(data_model.Company)
    )
   
    print(f"Calling Gemini for Company Info and using Yahoo Finance for stock price. {additional_instructions}")
    prompt=f"""
        Get company information for {ticker}. If the company is commonly known by another name, provide that name also.
        For example, for WW, the name would be "WW International Inc., formerly Weight Watchers International, Inc."
        {additional_instructions}
    """
    result = await agent.ainvoke(
        {
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
    )
    return result["structured_response"]
 

async def generate_company_overview(ticker: str, additional_instructions: str = "") -> data_model.CompanyOverview:
    """Generate a company overview such as the industry it is in, what the company does, and its key products."""
    print(f"Calling Gemini via Pydantic for Company Overview.  {additional_instructions}") 
    model = create_model()
    agent_without_tools = create_agent(
        model,
        system_prompt=f"""
        You are research analyst. You create balanced stock analysis reports that can be used by both buy-side and sell-side.
        Return structured JSON in the desired format.
        """,
        # Bug: with ProviderStrategy https://github.com/langchain-ai/langchainjs/issues/8585
        response_format=ToolStrategy(data_model.CompanyOverview)
    )
    prompt = f"""
    Provide a company overview of {ticker}: what industry it is in, what the company does, and its key products or segments.
    {additional_instructions}
    """
    result = await agent_without_tools.ainvoke(
        {
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
    )
    return result["structured_response"]


@tool
def download_webpage_content(url) -> str:
    """
    Downloads the HTML content of a webpage.
    """
    try:
        with urlopen(url) as response:
            # Read the content as bytes and then decode to a string
            content_bytes = response.read()
            content_string = content_bytes.decode('utf-8')  # Specify the correct encoding if known
            return content_string
    except Exception as e:
        print(f"Error downloading content: {e}")
        return ""
    

async def generate_key_financials(ticker: str, additional_instructions: str = "") -> data_model.KeyFinancialData:
    """Retrieve key financial data such as revenue and market-cap from up-to-date, trusted sources. """ 
    print(f"WARNING! Potentially hallucinated Key Financial Data.  {additional_instructions}")
    model = create_model(temperature=0.1)

    # Gemini's built-in tools do not seem to be supported in v1
    # https://docs.langchain.com/oss/python/integrations/providers/google
    # only lists API-based access to Google Search and no UrlContext
    # This documentation is wrong -- built-in web search seems to be there according to:
    # https://docs.langchain.com/oss/python/integrations/chat/google_generative_ai#built-in-tools

    agent_with_url = create_agent(
        model,
        # building a bespoke download tool runs into problems as the websites block bots: Too many requests
        # tools=[download_webpage_content],
        # Bug: with ProviderStrategy https://github.com/langchain-ai/langchainjs/issues/8585
        response_format=ToolStrategy(data_model.KeyFinancialData),
    )
    prompt =  f"""
        Get key financial data on {ticker} from https://finance.yahoo.com/quote/{ticker}/financials/
        Do not make up any numbers.
    """
    result = await agent_with_url.ainvoke(
        {
            "messages": [
                {"role": "user", "content": prompt}
            ]
        },
        tools=[GenAITool(url_context={})]
    )
    return result["structured_response"]


async def reformat_into_points(model, content) -> List[data_model.BulletPoint]:
    class Case(BaseModel):
        points: List[data_model.BulletPoint]

    format_prompt = f"""
        Convert the information below into the desired format.

        {content}
    """
    formatter_agent = create_agent(
        model,
        # Bug: with ProviderStrategy https://github.com/langchain-ai/langchainjs/issues/8585
        response_format=ToolStrategy(Case)
    )
    result = await formatter_agent.ainvoke(
        {
            "messages": [
                {"role": "user", "content": format_prompt}
            ]
        }
    )
    return result["structured_response"].points


async def generate_case(buy_side: bool, ticker: str, additional_instructions: str = "", n_points: int = 3) -> List[data_model.BulletPoint]:
    """Create talking points to make either a buy-side or sell-side case for the company""" 
    which_side = "buy-side" if buy_side else "sell-side"
    print(f"""Using Gemini's WebSearch tool for {which_side} points.  {additional_instructions}""")

    # built-in tools are not supported by the agent API, but are supported by the model API!
    model = create_model(temperature=0.1)
    prompt=f"""
        Search the web and find relatively recent analyst reports and summarize
        the key points to make the {which_side} case for the given stock.

        Make the {which_side} case for {ticker}. Produce exactly {n_points} points.
        Each title should be less than 10 words, and each point should be less than 200 words.

        {additional_instructions}
    """
    result = model.invoke(
        prompt,
        tools=[GenAITool(google_search={})]
    )
    points = await reformat_into_points(model, result.content)
 

    # Reflection (once)
    def validate(points) -> List[str]:
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
    
    error_messages = validate(points)
    if len(error_messages) == 0:
        return points


    # second attempt
    errors = '\n'.join(error_messages)
    print(f"Reflection: invoking model a second time to fix {len(error_messages)} errors")
    prompt2=f"""
        Search the web and find relatively recent analyst reports and summarize
        the key points to make the {which_side} case for the given stock.

        Make the {which_side} case for {ticker}. Produce exactly {n_points} points.
        Each title should be less than 10 words, and each point should be less than 200 words.

        Last time, you produced the following:
        {points}
        
        They had the following problems:
        {errors}
        Please fix.
    """
    result2 = model.invoke(
        prompt2,
        tools=[GenAITool(google_search={})]
    )
    points2 = await reformat_into_points(model, result2.content)
    error_messages2 = validate(points2)

    # return the better one
    if len(error_messages) < len(error_messages2):
        return points
    else:
        return points2


async def generate_analysis(ticker) -> data_model.StockReport:
    agent = create_deep_agent(
        model=create_model(),
        system_prompt=f"""
        You are an unbiased research analyst. You create balanced stock analysis reports that can be used
        by both buy-side and sell-side firms. 
        
        You have access to a number of tools that can retrieve  information on different topics from trusted topics.
        Use these tools to retrieve any information that you require.

        You should weave the information together into a coherent report. This might involve having to call
        a tool again based on newly discovered information. For example, if the buy-side case says that a company is
        experiencing growth in a product line, you should make sure that this product is listed in the key products
        and that the growth area is part of the company overview. Use the additional_instructions field in all the tools to
        call the tool again asking for such additional information.

        Finally, render the report in Markdown.       
        """,
        tools=[
            generate_company,
            generate_company_overview,
            generate_key_financials,
            generate_case,
            tools.render_report
        ]
    )

    prompt = f"Write a stock analysis report on {ticker}"
    result = await agent.ainvoke(
        {
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
    )
    return result["messages"][-1].content
 


async def main():
    """The main function."""
    parser = argparse.ArgumentParser(description="Create a stock analysis report.")
    parser.add_argument("ticker", help="The stock ticker symbol.")
    args = parser.parse_args()

    load_dotenv(dotenv_path="../keys.env") # GEMINI_API_KEY

    markdown_report = await generate_analysis(args.ticker)
    print(markdown_report)


if __name__ == "__main__":
    asyncio.run(main())

