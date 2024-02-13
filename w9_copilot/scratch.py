from langchain_google_vertexai import VertexAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.language_models.llms import LLM
from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string

MODEL_NAME = "gemini-pro"


def call_gemini_pro(model: LLM, prompt: str) -> str:
    result = ""
    for chunk in model.stream(prompt):
        print(chunk, end="", flush=True)
        result += chunk
    return result


def call_lcel(model: LLM, topic: str) -> str:
    prompt = ChatPromptTemplate.from_template("In the context of a W9 form, what does {topic} mean?")
    model = VertexAI(model_name=MODEL_NAME, temperature=0.8)
    output_parser = StrOutputParser()

    chain = prompt | model | output_parser
    return chain.invoke({"topic": topic})


def to_chat_history(messages: []) -> []:
    chat_history = []
    for message in messages:
        if message["role"] == "User":
            chat_history.append(HumanMessage(message["content"]))
        else:
            chat_history.append(AIMessage(message["content"]))
    return chat_history


def condense_question(messages: []) -> str:
    if len(messages) < 2:
        return messages[-1]['content']

    chat_history = to_chat_history(messages[:-1])
    prompt = PromptTemplate.from_template("""
        Given the following conversation and a follow up question,
        rephrase the follow up question to be a standalone question.
        
        ** Chat History **:
        {chat_history}
        
        ** Follow up question **:
        {question}
        
        ** Standalone question **:
    """)
    output_parser = StrOutputParser()
    chain = prompt | model | output_parser
    return chain.invoke({"chat_history": get_buffer_string(chat_history), "question": messages[-1]["content"]})


if __name__ == "__main__":
    model = VertexAI(model_name=MODEL_NAME)
    #print(call_gemini_pro(model, "What is the purpose of a W9 form?"))
    #print(call_lcel(model, "federal tax classification"))

    messages = [
        {"role": "User", "content": "What is the purpose of a W9 form?"},
        {"role": "Assistant", "content": "The purpose of the W9 form is to provide your correct taxpayer identification number to the person who is required to file an information return with the IRS."},
        {"role": "User", "content": "What does federal tax classification mean?"},
        {"role": "Assistant", "content": "Federal tax classification refers to the type of entity that is filing the W9 form."},
        {"role": "User", "content": "What are the possible values?"}
    ]
    print(condense_question(messages))