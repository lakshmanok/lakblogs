from google.adk.agents import Agent, LoopAgent
from google.adk.tools import google_search, url_context
from google.adk.tools import FunctionTool
from google.genai import types

import datetime
from . import tools, data_model
import os
import google.auth
from typing import List

WORKER_MODEL = "gemini-2.5-flash"

# Alternately, copy env.example to .env
# Get an API key from Google AI Studio and insert into .env
def setup_vertexai():
    _, project_id = google.auth.default()
    os.environ.setdefault("GOOGLE_CLOUD_PROJECT", str(project_id))
    os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")
    os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "True")

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
def create_company_agent():
    output_key = "Company"
    example = data_model.Company(name="WW International Inc., formerly Weight Watches International, Inc.",
                                 ticker="WW",
                                 latest_price=23.3)
    return Agent(
        model=WORKER_MODEL,
        name=f"{output_key}_agent",
        generate_content_config=create_model_settings(),
        description="Finds company info",
        instruction=f"""
        Get company information for the given TICKER. If the company is commonly known by another name, provide that name also.
        {example}
        """,
        output_key=output_key, # result will be in state[output_key]
        tools=[
            FunctionTool(tools.get_stock_price_mock),
        ],
    )
 

def create_company_overview_agent():
    output_key="CompanyOverview"
    print(f"Calling Gemini via Pydantic for {output_key}") 
    return Agent(
        model=WORKER_MODEL,
        name=f"{output_key}_agent",
        generate_content_config=create_model_settings(),
        description="Gets company overview",
        instruction=f"""
        Provide a company overview of TICKER -- what industry it is in, what the company does, and its key products or segments.
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
        instruction=f"""
        Get key financial data on TICKER from https://finance.yahoo.com/quote/TICKER/financials/
        Do not make up any numbers.
        Example:
        {example}
        """,
        output_key=output_key, # result will be in state[output_key]
        tools=[url_context]
    )


def create_case_creation_agent(buy_side: bool, n_points: int = 3):
    which_side = "buy_side" if buy_side else "sell_side"
    output_key = f"{which_side}_case"
    print(f"""Using Gemini's WebSearch tool for {which_side} points""")
    
    # verification tool
    def validate(points: List[data_model.BulletPoint]) -> List[str]:
        """Verifies a case (list of bullet points) and returns a list of error messages. """
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

    return Agent(
        model=WORKER_MODEL,
        name=f"{output_key}_agent",
        generate_content_config=create_model_settings(temperature=0.1),
        description=f"Creates a {which_side} case",
        instruction=f"""
        Your workflow is as follows:
        1. Search the web and find relatively recent analyst reports on the given company.
        2. Produce exactly {n_points} points making the {which_side} case for the given stock.
           Each point should have a title (of less than 10 words) and an explanation (of less than 200 words)
        3. Validate the list of bullet points and obtain set of error messages
        4. If there are errors, use the error messages as feedback to improve the generated case. Otherwise, return the current case.
        """,
        output_key=output_key, # result will be in state[output_key]
        tools=[
            google_search,
            FunctionTool(validate)
        ]
    )


# ADK is a fundamentally different architecture. It's an event loop.
def create_rootagent():
    stock_agent = Agent(
        model=WORKER_MODEL,
        name="final_report_agent",
        generate_content_config=create_model_settings(),
        description='Creates balanced stock analysis reports.',
        instruction=f"""
        You are an unbiased research analyst. You create balanced stock analysis reports that can be used
        by both buy-side and sell-side firms. 
        
        You have access to a number of sub-agents that can retrieve  information on different topics from trusted topics.
        Use these sub-agents to retrieve or generate any information that you require. Do NOT hallucinate this information.

        You should weave the information together into a coherent report. This might involve having to call
        a tool again based on newly discovered information. For example, if the buy-side case says that a company is
        experiencing growth in a product line, you should make sure that this product is listed in the key products
        and that the growth area is part of the company overview. Call the sub-agent again with the additional information.

        Your workflow is as follows:
        1. Ask the user to give you a stock symbol on which to create a report.
        2  Identify the ticker from the user query and provide it to any tools or agents that need the ticker being worked on.
        3. Look up Company Info corresponding to the ticker.
        4. Get a Company Overview for the ticker.
        5. Get key financial data for the ticker 
        6. Create a buy-side case for the company.
        7. Create a sell-side case for the company.
        8. Weave the information above together into a coherent report, rewriting sections for clarity and readability.
           You may have to call one of the subagents again to make the report consistent. For example, if the
           buy-side case says that a company is experiencing growth in a product line, you want to make sure that this
           product is listed in the key products section of the Company Overview. So, call the sub-agent again
           with this the additional information.
        9. Render the report in Markdown.
        10. Respond with the Markdown.

        Current date: {datetime.datetime.now().strftime("%Y-%m-%d")}
        """,
        output_key="final_report", # result will be in state["final_report"]
        tools=[
            FunctionTool(tools.render_report)
        ],
        sub_agents=[
            create_company_agent(),
            create_company_overview_agent(),
            create_key_financials_agent(),
            create_case_creation_agent(True),
            create_case_creation_agent(False)
        ]
    )
    return stock_agent

########################################
# Run (from parent directory) as: 
# adk run 4_adk
########################################
# setup_vertexai()  # commenting and using .env method
root_agent = create_rootagent() # has to be called root_agent
