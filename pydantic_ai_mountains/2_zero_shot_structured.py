import llm_utils
from pydantic_ai import Agent
from dataclasses import dataclass


@dataclass
class Mountain:
    name: str
    location: str
    height: float

if __name__ == '__main__':
    agent = Agent(llm_utils.default_model(),
                  result_type=Mountain,
                  system_prompt=(
                      "You are a mountaineering guide, who provides accurate information to the general public.",
                      "Provide all distances and heights in meters",
                      "Provide location as distance and direction from nearest big city",
                  ))

    questions = [
        "Tell me about the tallest mountain in British Columbia?",
        "Is Mt. Hood easy to climb?",
        "What's the tallest peak in the Enchantments?"
    ]

    for question in questions:
        print(">> ", question)
        answer = agent.run_sync(question)
        print(answer.data)

