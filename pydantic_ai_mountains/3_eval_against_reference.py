from typing import Tuple

import llm_utils
from pydantic_ai import Agent
from dataclasses import dataclass


@dataclass
class Mountain:
    name: str
    location: str
    height: float


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

    reference_answers = [
        Mountain("Robson", "Vancouver", 3954),
        Mountain("Hood", "Portland", 3429),
        Mountain("Dragontail", "Seattle", 2690)
    ]

    total_score = 0
    for l_question, l_reference_answer in zip(questions, reference_answers):
        print(">> ", l_question)
        l_answer = agent.run_sync(l_question)
        print(l_answer.data)
        l_score, l_reason = evaluate(l_answer.data, l_reference_answer)
        print(l_score, ":", l_reason)
        total_score += l_score

    avg_score = total_score / len(questions)
    print(f"Average score: {avg_score:.2f}")

