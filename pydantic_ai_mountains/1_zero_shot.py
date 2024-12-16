import llm_utils

if __name__ == '__main__':
    agent = llm_utils.agent()
    questions = [
        "What is the tallest mountain in British Columbia?",
        "Is it taller than Mt. Rainier?"
    ]

    print(">> ", questions[0])
    answer1 = agent.run_sync(questions[0])
    print(answer1.data)

    print(">> ", questions[1])
    answer2 = agent.run_sync(questions[1],
                             message_history=answer1.new_messages())
    print(answer2.data)
