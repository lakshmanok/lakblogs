# Comparing Agent Frameworks: PydanticAI, LangChain 1.0 and Google ADK

Six months ago, I was in favor of building using a low-level LLM abstraction library (Pydantic AI). We explicitly decomposed workflows into a sequence of LLM-calls one at a time, enabled the "agents" with task-specific tools and populated each one's context appropriately. We incorporating logging, guardrails, etc. into the LLM calls explicitly. That approach works, of course, and you can build complex LLM-based applications with it. The reason we chose that approach then was that agent frameworks were more trouble than they were worth. But this space moves fast, and it might be time to revisit some of that advice.

The release of LangChain 1.0 prompted us to take a second look at these frameworks. Is it a better choice? What do I want in an agent framework?
* It should simplify the building of production applications. (as opposed to toy demos)
* We should be able to easily collect datasets that we can use for evaluation and continuous training. This is a key value-proposition of Obin (my company! We are hiring. See open positions here: <https://www.obin.ai/careers/>).
* We should be able to run and test agents in isolation.
* We should be able to easily view traces, following a single user request or data preparation job end-to-end.
* We should be able to drop down to the model provider's unique capabilities (Google URL context grounding, Claude skills, etc.) at any time
* We should be able to take control of the agentic loop for consistent workflows.

Few low-code and no-code frameworks can provide us these capabilites. So, that leaves code-first frameworks like LangChain 1.0, Google ADK, Microsoft's AG2, and Crew AI. 

LangChain just did a massive ($125M) fund raising round for their Series B, and so should be around for a while. That said pre-1.0 LangChain was a mess in terms of conflicting documentation, and  layers of complexity that kept building over time. LangChain 1.0 is crisp and clean now, but it's still the same team. Will they get it right this time, or will the wide-ranging needs of the community make them lose their way again? 

I've been impressed by the velocity with which Google ADK is shipping. Obin is on Google Cloud, and we use Gemini quite extensively. So, Google ADK might be a natural choice. So, it's worth looking at.

Maybe I'll look at AG2 and CrewAI next weekend.

I'm take a simple, but realistic, use case and build it with the various options so that we can put these systems through their paces, and see how well these options meet our criteria.

## Non-agentic code
As a point of comparison and so that you understand what's happening in the framework examples, let's implement the use case with a non-agentic approach. Essentially, I'll populate the context with the necessary data and make an LLM call.

Given a stock symbol, we want to create a highly structured report with specific
sections. Let's create a data model representing what we want
(full code is at <https://github.com/lakshmanok/lakblogs/blob/main/stockanalysis/data_model.py>)
```
from typing import List
from pydantic import BaseModel, Field

class Company(BaseModel):
    name: str
    ticker: str
    latest_price: float = Field(description="The latest price of the stock.")

...

class StockReport(BaseModel):
    company: Company
    company_overview: CompanyOverview
    key_financial_data: KeyFinancialData
    buy_case: List[BulletPoint]
    sell_case: List[BulletPoint]
```

In a non-agentic application, we simply prompt the LLM -- there are no tools,
self-checks, or sub-agents. Using the Gemini client API, this is what that looks like:
(full code is in <https://github.com/lakshmanok/lakblogs/blob/main/stockanalysis/1_non_agentic.py>):
```
    price = await tools.get_stock_price(ticker) 
    prompt = f"Create a stock analysis report for {ticker}. The current price is ${price:.2f}."
    client = genai.Client()
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={
            "response_mime_type" : "application/json",
            "response_schema" : StockReport,
        }
    )
    obj: StockReport = response.parsed
```

Note a few things:
* Because the stock price is a real-time value, we call a function to get the current value and add it to the context. The point of a tool is that the LLM decides when to call it, but if you know it's always needed, why bother with all that? You can just do stuff.
* We are using Gemini's Structured Outputs functionality to get back a precise Pydantic object. This is an essential part of building robust production applications because downstream classes and functions can rely on the data structure. Dicts, JSON, etc don't give you this -- data classes help you write code that is not littered with error checks.

The code is pretty straightforward, and simple, but there are issues with the report as generated:
* The price is correct because we passed it in, but the key financial data might be old or hallucinated.
* The data structure says that we want a list of bullet points. In the tearsheets commonly used in finance, there is often a fixed space allowed and so we need more precise control on the bullet points in the buy-side and sell-side cases. We can not easily do that by defining just an output schema.

Let's fix both of these with an agentic approach. We'll start with the composable agents approach that we currently follow at Obin, with Pydantic AI as our LLM framework.

## Composable approach with Pydantic AI
It's worth discussing about what a composable approach to this problem looks like. You'd assign one agent each to every section of the report. The Company agent would be armed with a tool call to get price. The Key Financials agent might be given a specific URL to scrape the data from. The buy-side and sell-side agents might be armed with a Google Search tool to pull down the latest analyst reports and summarize them. Every prompt will get created by a templatized Prompt Service that uses Jinja2. You would log the prompts in this service, and make sure to log all AI generated responses for evaluation. Then, you'd use the logging configuration to ensure that prompts go to a different log file than evals.

You can see an example of this approach in the [GitHub repository](https://github.com/lakshmanok/generative-ai-design-patterns/blob/main/composable_app/agents/generic_writer_agent.py#L74) of our book Generative AI Design Patterns (O'Reilly, 2025):
<pre>
async def revise_article(self, topic: str, initial_draft: Article, panel_review: str) -> Article:
        # the prompt is the same for all writers
        prompt_vars = {
            "prompt_name": "AbstractWriter_revise_article",
            "topic": topic,
            "content_type": self.get_content_type(),
            "additional_instructions": ltm.search_relevant_memories(f"{self.writer.name}, revise {topic}"),
            "initial_draft": initial_draft.to_markdown(),
            "panel_review": panel_review
        }
        prompt = PromptService.render_prompt(**prompt_vars) # this logs prompt
        result: Article = await self.revise_response(prompt)
        await evals.record_ai_response("revised_draft", # you'd use this to filter
                                       ai_input=prompt_vars, # useful for simulation
                                       ai_response=result) # this logs response
        return result
</pre>

You can see the explicit use of long-term memory (ltm), Jinja templating of prompts (PromptService), and logging for evals. Please do refer to that repository if you want to see what these layers look like. I'll skip them here in the interest of keeping the code simple and easy to understand.

### Pydantic AI with Subagents
The full code implementing the stock report workflow with Pydantic AI is at <https://github.com/lakshmanok/lakblogs/blob/main/stockanalysis/2b_pydanticai_subagents.py>. To keep the example concise, I've not added the PromptService, llm wrapper, evals, etc. that we'd want to use in production.

Let's walk backwards. To generate the final report, we'll basically execute a number of independent LLM calls (subagents) in parallel (possible because they are independent):
```
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
```

Let's look at each of these subagents because they illustrate different aspects of a realistic agent workflow and help us answer the questions that I started this article with.

### Tool Calling and Structured Outputs
The company information includes the current price of the stock. The agent can be armed with a tool call to call the function that gets the price:
```
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
        return await tools.get_stock_price(ticker)
    
    print("Calling Gemini via Pydantic for Company Info and using Yahoo Finance for stock price")
    result = await agent.run(
        f"""
        Get company information for {ticker}. If the company is commonly known by another name, provide that name also.
        For example, for WW, the name would be "WW International Inc., formerly Weight Watchers International, Inc."
        """,
        output_type=data_model.Company
    )
    return result.output
```

A few things to note:
* Pydantic allows us to break the LLM abstraction if needed and get to the original LLM. We are using it to turn on Gemini's built-in guardrails at an appropriate level. This will save us latency over trying to do Guardrails via postprocessing (this is a common theme: latency is the bug-bear of production applications, and you want to do as much as possible server-side without an intervening network call)
* Along with the system prompt, we can specify retries when creating the agent. Pydantic takes care of this for us.
* The code illustrates specifying a function tool (get_stock_price) and an output schema (data_model.Company).  This appears to be basic stuff, but you'll see LangChain and ADK struggle to do this cleanly.

### Scraping a URL to get real-time data + workaround Gemini limitation
Let's say that we want to get actual financial data from Yahoo Finance.
We could download the website, and pass the text content to the model within the
context. But this is surprisingly error-prone in production applications.
Many websites block or throttle bots. Ads, popups, etc. also make it likely that you will not be able to properly parse the page.

A good solution is to use Gemini's UrlContext feature. Essentially, Google uses their own index of the website to populate the context for you. Not only is there a latency benefit, you don't have to deal with things like ads and popups.

Now, only Gemini does this. At this time, Claude/GPT-5 don't. But Pydantic AI doesn't try to make a generic LLM wrapper, so they do have a UrlContextTool which will tell Gemini to do its thing:
```
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
        output_type=PromptedOutput(data_model.KeyFinancialData) # Gemini does not support output tools and built-in tools at the same time. Use `output_type=PromptedOutput(data_model.KeyFinancialData)
    )
    return result.output
```

There is one point above that you have to be aware of. This is going to trip us.

Google, probably because this was built by different teams that don't talk to each other, makes you choose between Structured Outputs and Built-in Tools. So, if you use UrlContext, you don't get to say that you want the financial data in a structured dataclass. Yikes!

Fortunately, Pydantic AI gives you a workaround -- they will take the output and format it for you using the LLM in the structure you want. That's what the PromptedOutput does.

### Reflection
The buy-side and sell-side case generation shows us using Reflection, a key aspect of many agents.

The idea is that we use Google Search to find analyst reports and get a list of bullet points:
```
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
```

Then, we validate the result, checking that it does fit the specifications:
```
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
```
Because this is just Python, it's as simple as calling the method on the returned value. Again, seemingly simple stuff, but you'll see LangChain and ADK struggle with this.

Then, we invoke the agents with additional instructions formed from the error messages and get them to correct their response.


### Limitations of low-level LLM abstraction
I like Pydantic AI because it gives me a low-level LLM abstraction without getting in the way too much. The limitations of this approach started to become evident as we started to build more and more at Obin. It is fine for small projects (and allows an individual developer to move fast), but breaks down for larger teams and as you start to use LLM-specific capabilities:

What limitations?
* Abstractions are imperfect: The same feature (such as structured outputs) can be implemented very differently. Claude does post-formatting, so has twice the latency whereas Gemini zeroes out the logits and so does so with a single class. The tradeoff is the Claude has a lower refusal rate. 
* Abstractions give you the lowest common denominator: The LLM providers' API is often much richer than what the abstraction of the world expose. For example, the Claude SDK supports Excel handling, financial data through MCP, has prompt improvement capabilities, and now has skills. Gemini has very good media caching, Google Search-based grounding, the ability to load URLs into the context, a sandbox to run generated code, etc. You often lose these or lag significantly behind when you use an abstraction layer over the LLM.
* Composability complicates context curation. You don't want to keep appending things to context; you often need to curate them, and this requires knowledge of what the downstream agents still need. 
* Composability can complicate testing unless you take care to design every agent as being stateless. It's hard to test agents individually when they depend on the context having been formed properly by upstream agents. On the other hand, stateless agents might waste tokens and network bandwidth if you don't take the effort to curate context.

Ultimately, best practices are unenforceable as teams grow. For example, it is hard to enforce proper use of the PromptService, evals logging, etc. as team size increases. New developers don't realize why these wrappers exist, directly invoke the underlying service, and then you start to have holes in your logs and guardrails.

Now, let's see whether LangChain and ADK make this easier, and how.


## LangChain 1.0
Let's build the same use case with LangChain.  You can see the full code at <https://github.com/lakshmanok/lakblogs/blob/main/stockanalysis/3_langchain.py>

As before, let's work backwards. In LangChain, as in PydanticAI, you can compose the report by running a bunch of subagents in parallel:
```
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
```
The immaturity of LangChain 1.0 starts to show up when you start to look at real-world needs.

### No async tools
Take the simplest thing. All tool calls have to be synchronous. I had to do:
```
@tool
def get_stock_price(ticker: str) -> float:
    """Get the latest closing stock price of a publicly traded stock."""
    return asyncio.run(
        # no way to call async tools!
        tools.get_stock_price(ticker)
    )
```
Given that tool calls in most LLM applications involve network calls, this is a suprising miss.

### Native structured outputs not supported
Even in cases where Gemini supports Structured Outputs, LangChain fails. I found that I had to use ToolStrategy (which involves reformatting the output response) rather than ProviderStrategy all the time:
```
    agent_without_tools = create_agent(
        model,
        system_prompt=f"""
        You are research analyst. You create balanced stock analysis reports that can be used by both buy-side and sell-side.
        Return structured JSON in the desired format.
        """,
        # Bug: with ProviderStrategy https://github.com/langchain-ai/langchainjs/issues/8585
        response_format=ToolStrategy(data_model.CompanyOverview)
    )
```

### UrlContext is not supported
Remember the really cool built-in Gemini feature where they serve webpage content out of their index? Well, you can't use it in LangChain. <https://docs.langchain.com/oss/python/integrations/providers/google> only lists API-based access to Google Search and no UrlContext.

Trying to download webpage content runs into all the limitations I mentioned earlier.

### Conflicting capabilities between model and agent
In the previous section, I said that the website that lists integrations with Google does not list the Google Search tool that's built-in to Gemini. Instead, only an API-based Google Search is listed.

Well it turns out that those are the integrations for agents. If you use the model, you can get the built-in tool.
```
result = model.invoke(
        prompt,
        tools=[GenAITool(google_search={})]
    )
```
The documentation is confusing. The integration with agents is not the only way to access model capabilities, so there should be some hint that you can drop down to the model. I suspect that this distinction without a difference is going to be cause of many a few wasted hours.

But as, with Pydantic, the ugly Google error that built-in tools and Structured Outputs don't mix rears its head. Where Pydantic gives you a helpful workaround, LangChain leaves you to figure out what's happening (it's not easy) and solve it yourself.  I had to write a reformatter:
```
async def reformat_into_points(model, content) -> List[data_model.BulletPoint]:
    class Case(BaseModel):
        points: List[data_model.BulletPoint]

    format_prompt = f"""
        Convert the information below into the desired format.

        {content}
    """
    formatter_agent = create_agent(
        model,
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

```

LangChain pre-1.0 was notorious for confusing layers of abstraction and open issues that were hard to troubleshoot. So, this is not a good sign.

Overall, it took me 176 lines to implement the subagent workflow in Pydantic AI,
and 266 lines to do so in LangChain 1.0. So, they are comparable. 

### Advantages of LangChain: Observability, Deep Agents
One big advantage with building with LangChain is the ability to incorporate middleware and horizontal capabilities easily. See <https://docs.langchain.com/oss/python/langchain/middleware>. You can boostrap logging for evals onto agents through middleware, but it is still one agent at a time. Tracing can be accomplished through LangSmith and integration with Arize Phoenix is straightforward.

Of course, LangChain is not *meant* to be used when you want such precise control. Creating precisely formatted stock analysis reports with non-hallucinated data is not the use case that they are going after -- they are going after use cases where you can give up quite a bit of the planning and execution responsibility to the LLM.

What this means in practice is that, rather than orchestrating the report explicitly in code, you might use "Deep Agents", telling it to use sub agents as needed:
```
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
```

The key benefit here is how it was to ask the final agent to *weave* together the information into a coherent report and go back and ask *any* agent to redo its work given the work.  If the definition of an agentic application is that you can call tools and subagents in a loop, this final change is what makes the application fully agentic.

The full code for a deep agent implementation is at <https://github.com/lakshmanok/lakblogs/blob/main/stockanalysis/3c_langchain_deepagents.py>

## Google ADK
Let's build the same use case with Google ADK. The full code for this section is at <https://github.com/lakshmanok/lakblogs/blob/main/stockanalysis/4_adk/agent.py>

### It's a chatbot and it has an event loop
The first thing to realize is that Google ADK is *very* opionionated. By default, they expect you to build a conversational application. You can build a non-conversational application, but then you have to create a session, runner, etc. and populate it with nonsensical user-ids and session-ids (and remember what they are, because the internal state is tracked by user and session).

Let's not fight them. As before, let's walk backwards. Here, we want a sequential pipeline of sub-agent calls:
```
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
 
# To run this program:   adk run 4_adk
root_agent = create_pipeline_agent() # has to be called root_agent
```

Note that, unlike with Pydantic AI or LangChain, we are just constructing agents. We are *not* invoking the agents. That's because ADK runs everything in an event loop (like Streamlit or Spring) and it takes care of invoking the agents.
This also means you can not just run the main(). Instead you launch the chat application as:
```
adk run <parentdir>
```
There is also a web interface to the chat application which gives you a lot of bells and whistles. It's really nice.

### User input into a state variable
However, because it's a chat application, you see that I can not just pass the ticker into the application. No siree. You can figure out how to add something to the session state that is managed by the event loop (good luck), or take the simpler way out of taking the user question and extracting the ticker symbol from it:
```
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
```

You notice that I've made lemonade from lemons. If we are going to extract the ticker from the user input, we might as well make it more powerful and allow the user more leeway.

ADK is opionionated, but its opinions are usually good ones. It can, however, get frustrating to figure out what the intended path is when the obvious path of passing in a ticker to the first agent is closed off.

### Structured Outputs go away
Surprisingly, ADK has not figured out how to marry tool calling and structured outputs. You pretty much has to forget having clean contracts between steps of your workflow and hope that the LLM can reformat the outputs of one step into the inputs of the next step.

What I found helpful was to create examples and use those examples as JSON to help guide the LLM on what the contract of the next step was:
```
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
```

Because the output key is "ticker", I can use "{ticker}" within any prompt and this variable will get injected into downstream agents.

Of course, this also means that you can not use f-strings in Python for your prompts because that's the same exact syntax that Python expects. Not sure what the ADK designers were thinking. This explains the weird mixture of non-f-string and f-string in the instruction parameter above. Good bye code readability and maintainability.

## Reflection
As with LangChain, I had to explicitly format the output of the agent in order to validate it, but the presence of higher-level control structures like LoopAgent means that reflection can be done quite easily:
```
    case_creation_agent = LoopAgent(
        name=f"{output_key}_loop_agent",
        max_iterations=2,
        sub_agents=[
            case_creation_agent,
            formatter_agent,
            validation_agent,
        ]
    )
```

The Google ADK implementation runs to 268 lines (Pydantic was 176 and LangChain was 266). Surprisingly, the higher-level abstraction did not result in fewer lines of code. But that may be because I had to write a couple of extra agents and longer prompts.

Ultimately, Google ADK gives me TensorFlow 1.0 vibes. It's written by engineers who are far better than me for people like them and who have their needs. The event loop paradigm reminds me of the struggles with data reading mechanism (feed_dict, placeholder, etc.) -- it was very scalable but no one quite understood it and by the time Google listened to feedback and implemented tf.data, the world had moved on.  I suspect that Google ADK will have the same fate. 

Google builds great tools -- as you can see from this post, I really love Gemini's capabilities, and its built-in tools like context caching, search tool, UrlContext, etc. are incredibly effective in production applications. But they seem to consistently drop the ball when it comes to developer frameworks. The tools (and LLM) are great because they are tested at Google scale. Unfortunately, it's a double-edged sword. The developer framework is hard to understand and use because the needs of internal Google are very unlike the needs of everyday applications.

## Summary Thoughts
LangChain 1.0 is very promising:
* The middleware hooks will enable a large ecosystem of tools to hooking into LangChain. We already see this with observability (Arize), guardrails (Guardrails AI), etc. 
* The community is building frameworks for deep agents, deep research, etc. Like HuggingChain for open-weights models, I can see LangChain becoming the defacto place that new OSS agentic capabilities are found.
* It's comparable to Pydantic AI in its ability to support precise control of workflows.
* That precise control does not come at the expense of building more autonomous applications where the LLM can itself choose which subagent or tools to call.

LangChain 1.0 is still immature:
* The LLM abstraction is spotty. You are meant to be able to use the high-level abstraction, but still be able to drop down to the model if necessary. But I found several key capabilities of the underlying LLM missing. I am using Gemini, but I suspect that anyone using Claude in anger with LangChain will find analagous limitations.
* Tool calls have to be synchronous, and so this will add latency to many applications that wrap remotely hosted APIs via tool calls.
* There is a lot of confusion between model and agent in the API. This is likely to get worse as more capabilities get added. Seriously, they should have copied or adopted Pydantic AI's clean API surface instead of trying to invent their own.

Between LangChain 1.0 and Google ADK, I'd choose LangChain. If, like us, you use Pydantic AI today, I'd wait a couple of months before considering LangChain. If you are not using an agent framework today, and want to build high-level agents, use LangChain 1.0. But prepare to be frustrated.

<em> 
Lak is co-founder and CTO of Obin.ai which is building deep domain agents for finance.
We are hiring. Please see open positions here: <https://www.obin.ai/careers/> 
</em>
