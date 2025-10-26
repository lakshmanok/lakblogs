from google.adk.agents import Agent, LoopAgent, SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search, url_context
from google.adk.tools import FunctionTool
from google.genai import types

import datetime
from . import tools, data_model
import os
import google.auth
from typing import List
from pydantic import BaseModel

import argparse
import asyncio

WORKER_MODEL = "gemini-2.5-flash"

# Alternately, copy env.example to .env
# Get an API key from Google AI Studio and insert into .env
def setup(vertexai=False):
    if vertexai:
        _, project_id = google.auth.default()
        os.environ.setdefault("GOOGLE_CLOUD_PROJECT", str(project_id))
        os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")
        os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "True")
    else:
        
        os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "False")

def create_model_settings(temperature: float = 0.25):
    return types.GenerateContentConfig(
        temperature=temperature,
        safety_settings=[
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH
            )
        ]
    )

# we only create the Agents. the agents are invoked by ADK's event loop
def extract_ticker_agent():
    output_key = "ticker"
    print(f"Calling Gemini for {output_key}") 
    return Agent(
        model=WORKER_MODEL,
        name=f"{output_key}_agent",
        generate_content_config=create_model_settings(),
        description="Finds ticker",
        instruction=f"""
        From the user query, extract the stock market symbol that the user wants a report on.
        <example>
        User: Please write a report on Microsoft
        AI:   MSFT
        </example>
        <example>
        User: Analyze WW
        AI:   WW
        </example>
        <example>
        User: Nvidia
        AI:   NVDA
        </example>
        """,
        output_key=output_key, # result will be in state[output_key]
    )

def create_company_agent():
    output_key = "Company"
    print(f"Calling Gemini for {output_key}") 
    example = data_model.Company(name="WW International Inc., formerly Weight Watches International, Inc.",
                                 ticker="WW",
                                 latest_price=23.3)
    return Agent(
        model=WORKER_MODEL,
        name=f"{output_key}_agent",
        generate_content_config=create_model_settings(),
        description="Finds company info",
        instruction="""
        Get company information for {ticker}.
        If the company is commonly known by another name, provide that name also.
        """ + f"Example output: {example.model_dump_json()}",
        output_key=output_key, # result will be in state[output_key]
        tools=[
            FunctionTool(tools.get_stock_price_mock),
        ],
    )
 

def create_company_overview_agent():
    output_key="CompanyOverview"
    print(f"Calling Gemini for {output_key}") 
    return Agent(
        model=WORKER_MODEL,
        name=f"{output_key}_agent",
        generate_content_config=create_model_settings(),
        description="Gets company overview",
        instruction="""
        Provide a company overview of {ticker} -- what industry it is in, what the company does, and its key products or segments.
        """,
        output_key=output_key, # result will be in state[output_key]
    )


def create_key_financials_agent():
    output_key="KeyFinancialData"
    print(f"Using Gemini's UrlContext for {output_key}")
    example = data_model.KeyFinancialData(
        market_cap = "$300M",
        revenue = "$80M",
        net_income = "($34.5M)",
        pe_ratio = "N/A",
        dividend_yield="0"
    )
    return Agent(
        model=WORKER_MODEL,
        name=f"{output_key}_agent",
        generate_content_config=create_model_settings(temperature=0.1),
        description="Gets key financial data",
        instruction="""
        Get key financial data on TICKER from https://finance.yahoo.com/quote/{ticker}/financials/
        Do not make up any numbers.""" + f"Example: {example.model_dump_json()}",
        output_key=output_key, # result will be in state[output_key]
        tools=[url_context]
    )


def create_case_creation_agent(buy_side: bool, n_points: int = 3):
    which_side = "buy_side" if buy_side else "sell_side"
    output_key = f"{which_side}_case"
    print(f"""Using Gemini's WebSearch tool for {which_side} points""")

    case_creation_agent = Agent(
        model=WORKER_MODEL,
        name=f"{output_key}_agent",
        generate_content_config=create_model_settings(),
        description=f"Creates a {which_side} case",
        instruction=f"Make the {which_side} case for " + "{ticker}" + f"""
        Your workflow is as follows:
        1. Search the web using Google Search and find relatively recent analyst reports on the given company.
        2. Produce exactly {n_points} points making the {which_side} case for the given stock.
        3. Incorporate feedback (error messages) from previous runs, if any.
        """,
        output_key=output_key, # result will be in state[output_key]
        tools=[
            google_search
        ]
    )

    class Case(BaseModel):
        points: List[data_model.BulletPoint]

    formatter_agent = Agent(
        model=WORKER_MODEL,
        name=f"{output_key}_formatter_agent",
        generate_content_config=create_model_settings(),
        description=f"Formats the {which_side} case",
        instruction="""
        Reformat or rewrite the bullet points below into a structure where 
        each point has a title (of less than 10 words) and an explanation (of less than 200 words)
        
        """ + "{" + output_key + "}\n",
        output_key=f"{output_key}_formatted", # result will be in state[output_key]
        output_schema=Case
    )

    # verification tool
    def validate(case: Case) -> List[str]:
        """Verifies a case (list of bullet points) and returns a list of error messages. """
        points = case.points
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
    
    validation_agent = Agent(
        model=WORKER_MODEL,
        name=f"{output_key}_validator_agent",
        generate_content_config=create_model_settings(),
        description=f"Validates a {which_side} case",
        instruction=f"""
        Validate the list of bullet points and obtain set of error messages using the given tool, which returns a list of error messages, if any.
        If there are errors, use the error messages as feedback to improve the generated case. Otherwise, return the generated bullet points.
        
        """ + "{" + output_key + "_formatted}\n",
        output_key=f"{output_key}_error_messages", # result will be in state[output_key]
        tools=[
            FunctionTool(validate)
        ]
    )

    case_creation_agent = LoopAgent(
        name=f"{output_key}_loop_agent",
        max_iterations=2,
        sub_agents=[
            case_creation_agent,
            formatter_agent,
            validation_agent,
        ]
    )

    return case_creation_agent

    

def create_final_report_agent():
    output_key = "final_report"
    print(f"""Assembling report from state variables""")
    return Agent(
        model=WORKER_MODEL,
        name=f"{output_key}_agent",
        generate_content_config=create_model_settings(),
        description="Assembles report",
        instruction="""
        Using the following information, construct a complete report
        in the desired JSON format.

        Company:
        {Company}

        CompanyOverview:
        {CompanyOverview}

        KeyFinancials:
        {KeyFinancialData}

        buy-side case:
        {buy_side_case}

        sell-side-case:
        {sell_side_case}
        """,
        output_key=output_key, # result will be in state[output_key]
        output_schema=data_model.StockReport
    )

# ADK is a fundamentally different architecture. It's an event loop.
def create_pipeline_agent():
    final_pipeline_agent = SequentialAgent(
        name="pipeline_agent",
        description='Creates balanced stock analysis reports using a specific set of steps.',
        sub_agents=[
            # will run in this order
            extract_ticker_agent(),
            create_company_agent(),
            create_company_overview_agent(),
            create_key_financials_agent(),
            create_case_creation_agent(True),
            create_case_creation_agent(False),
            create_final_report_agent()
        ]
    )
    return final_pipeline_agent

########################################
# If you create the root_agent here, you can run interactive session (from parent directory) as: 
# adk run 4_adk
root_agent = create_pipeline_agent() # has to be called root_agent
########################################
