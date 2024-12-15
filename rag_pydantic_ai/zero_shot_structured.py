import llm_utils
from pydantic_ai import Agent
from dataclasses import dataclass
from pydantic import Field

@dataclass
class Mountain:
    name: str
    height: float # in meters
    taller_than_mt_rainier: bool


if __name__ == '__main__':
    agent = Agent(llm_utils.default_model(), result_type=Mountain)
    question = "What is the tallest mountain in British Columbia? How tall is it, in meters? Is it taller than Mt. Rainier?"
    print(">> ", question)
    answer1 = agent.run_sync(question)
    print(answer1.data)

