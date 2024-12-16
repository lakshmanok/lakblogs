from typing import Tuple
import llm_utils
from pydantic_ai import Agent, RunContext
from dataclasses import dataclass
import wikipedia_tool
from pydantic_ai_mountains.wikipedia_tool import OnlineWikipediaContent, FakeWikipediaContent


@dataclass
class Mountain:
    name: str
    location: str
    height: float

class Tools:
    elev_wiki: wikipedia_tool.WikipediaContent
    def __init__(self):
        self.elev_wiki = OnlineWikipediaContent("List of mountains by elevation")

class FakeTools:
    elev_wiki: wikipedia_tool.WikipediaContent
    def __init__(self):
        self.elev_wiki = FakeWikipediaContent("List of mountains by elevation")


def evaluate(answer: Mountain, reference_answer: Mountain) -> Tuple[float, str]:
    score = 0
    reason = []
    if reference_answer.name in answer.name:
        score += 0.5
        reason.append("Correct mountain identified")
        if reference_answer.location in answer.location:
            score += 0.25
            reason.append("Correct city identified")
        height_error = abs(reference_answer.height - answer.height)
        if height_error < 10:
            score += 0.25 * (10 - height_error)/10.0
        reason.append(f"Height was {height_error}m off. Correct answer is {reference_answer.height}")
    else:
        reason.append(f"Wrong mountain identified. Correct answer is {reference_answer.name}")

    return score, ';'.join(reason)


if __name__ == '__main__':
    # Create agent, and arm it with capabilities
    agent = Agent(llm_utils.default_model(),
                  result_type=Mountain,
                  system_prompt=(
                      "You are a mountaineering guide, who provides accurate information to the general public.",
                      "Use the provided tool to look up the elevation of many mountains."
                      "Provide all distances and heights in meters",
                      "Provide location as distance and direction from nearest big city",
                  ))
    @agent.tool
    def get_height_of_mountain(ctx: RunContext[Tools], mountain_name: str) -> str:
        return ctx.deps.elev_wiki.snippet(mountain_name)


    # Now try it out
    questions = [
        "Tell me about the tallest mountain in British Columbia?",
        "Is Mt. Hood easy to climb?",
        "What's the tallest peak in the Enchantments?"
    ]

    reference_answers = [
        Mountain("Robson", "Vancouver", 3954),
        Mountain("Hood", "Portland", 3429),
        Mountain("Dragontail", "Seattle", 2690)
    ]

    tools = FakeTools()  # Tools or FakeTools

    total_score = 0
    for l_question, l_reference_answer in zip(questions, reference_answers):
        print(">> ", l_question)
        l_answer = agent.run_sync(l_question, deps=tools) # note how we are able to inject
        print(l_answer.data)
        l_score, l_reason = evaluate(l_answer.data, l_reference_answer)
        print(l_score, ":", l_reason)
        total_score += l_score

    avg_score = total_score / len(questions)
    print(f"Average score: {avg_score:.2f}")

