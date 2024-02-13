from langchain_google_vertexai import VertexAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.llms import LLM

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


if __name__ == "__main__":
    model = VertexAI(model_name=MODEL_NAME)
    print(call_gemini_pro(model, "What is the purpose of a W9 form?"))
    print(call_lcel(model, "federal tax classification"))
