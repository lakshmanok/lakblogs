## Comparing agentic approaches using LangChain 1.0 and Google ADK

Six months ago, I was in favor of building using a low-level LLM abstraction library (Pydantic AI). We explicitly decomposed workflows into a sequence of LLM-calls one at a time, enabled the "agents" with task-specific tools and populated each one's context appropriately. We incorporating logging, guardrails, etc. into the LLM calls explicitly. That approach works, of course, and you can build complex LLM-based applications with it. The reason we chose that approach then was that agent frameworks were more trouble than they were worth. But this space moves fast, and it might be time to revisit some of that advice.

The limitations of the composable agents + LLM abstraction approach started to become evident as we started to build Obin:
* Abstractions are imperfect: The same feature (such as structured outputs) can be implemented very differently. Claude does post-formatting, so has twice the latency whereas Gemini zeroes out the logits and so does so with a single class. The tradeoff is the Claude has a lower refusal rate. 
* Abstractions give you the lowest common denominator: The LLM providers' API is much richer than what the Pydantic etc. of the world expose. For example, the Claude SDK supports Excel handling, financial data through MCP, has prompt improvement capabilities, and now has skills. Gemini has very good media caching, Google Search-based grounding, the ability to load URLs into the context, a sandbox to run generated code, etc. You lose these when you use an abstraction layer over the LLM. 
* Composability complicates context curation. You don't want to keep appending things to context; you often need to curate them, and this requires knowledge of what the downstream agents still need.
* Composability complicates testing. It's hard to test agents individually when they depend on the context having been formed properly by upstream agents.
* Best practices are unenforceable. For example, it is hard to enforce proper use of decorators as team size increases. New developers don't realize why these decorators exist, directly invoke the underlying service, and then you start to have holes in your logs and guardrails.

The release of LangChain 1.0 and the rapid shipping speed of Google ADK prompted us to take a second look at these frameworks. Are they a better choice? What do I want in an agent framework?
* It should simplify the building of production applications. (as opposed to toy demos)
* We should be able to easily collect datasets that we can use for evaluation and continuous training. This is a key value-proposition of Obin.
* We should be able to run and test agents in isolation.
* We should be able to easily view traces, following a single user request or data preparation job end-to-end.
* We should be able to drop down to the model provider's unique capabilities (Google URL context grounding, Claude skills, etc.) at any time
* We should be able to take control of the agentic loop

Few low-code and no-code frameworks can provide us these capabilites. So, that leaves code-first frameworks like LangChain 1.0, Google ADK, Microsoft's AG2, and Crew AI. According to my sources, LangChain and Google ADK are #1 and #2 right now in terms of usage, with LangChain obviously orders of magnitude more popular. 

LangChain just did a massive ($125M) fund raising round for their Series B, and so should be around for a while. That said pre-1.0 LangChain was a mess in terms of conflicting documentation, and  layers of complexity that kept building over time. LangChain 1.0 is crisp and clean now, but it's still the same team. Will they get it right this time, or will the wide-ranging needs of the community make them lose their way again? 

Google ADK has been seeing incredible shipping velocity, and internal teams at Google are leaning leavily into it. However, there is a tension between the ease-of-use that Deep Mind & Cloud need and scale-to-billion-user-products that internal Google requires. Historically, Google has not handled this well (Tensorflow). Colab and Keras show that Google can get this tradeoff right. However, it's hard to trust Google when it comes to developer tools -- ADK might be cut at the whim of the next executive hire (remember Google Stadia that Google swore up-and-down was a long-term commitment?). 

I'll break a tie or even a close result in favor of LangChain.

I'm take a simple, but realistic, use case and build it with the various options so that we can put these systems through their paces, and see how well these options meet our criteria.

## Non-agentic code
As a point of comparison and so that you understand what's happening in the framework examples, let's implement the use case with a non-agentic approach. Essentially, I'll populate the context with the necessary data and make an LLM call.

## Composable approach
It's worth thinking about what a composable approach here looks like. You'd assign one agent to each section of the report. The company summary agent would be armed with a tool call to get price. The buy-side and sell-side agents might be armed with a Google Search tool to validate the data. Every call will go though llms.py and be decorated through logging to ensure that you get the input and output prompts, and you'd use the logging configuration to ensure that prompts go to a different log file than evals.

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
        prompt = PromptService.render_prompt(**prompt_vars)
        result: Article = await self.revise_response(prompt)
        await evals.record_ai_response("revised_draft",
                                       ai_input=prompt_vars,
                                       ai_response=result)
        return result
</pre>

You can see the explicit use of long-term memory (ltm), Jinja templating of prompts (PromptService), and logging for evals. In a short example, I don't want to have to build the long-term memory, logging for evals, etc. so it's necessarily incomplete, but we can see what the subagents approach with Pydantic AI would look like.


As I said, the limitations of this approach started to become evident as we started to build Obin. It is fine for small projects (and allows an individual developer to move fast), but breaks down for larger teams and as you start to use LLM-specific capabilities.

Now, let's see whether LangChain and ADK make this easier, and how.  I'll do both the monolithic application and the more realistic sub-agents approach.

## LangChain 1.0
Let's build the same use case with LangChain


One big advantage with building with LangChain is the ability to incorporate middleware and horizontal capabilities easily.
https://docs.langchain.com/oss/python/langchain/middleware

## Google ADK
Let's build the same use case with Google ADK

## Summary Thoughts
