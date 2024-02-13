import streamlit as st
import pandas as pd
import urllib.request
import os
import base64

from langchain_core.messages import HumanMessage, AIMessage, get_buffer_string
from langchain_google_vertexai import VertexAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.language_models.llms import LLM


def ask_question(model: LLM, messages: [{}]) -> str:
    question = condense_question(messages)

    prompt = ChatPromptTemplate.from_template("""
        You are a helpful tax advisor and will answer questions at a sixth-grade reading level.
        Keep responses shorter than 3 sentences.
        Your goal is to help the person asking the question complete the W9 form correctly.
        Assume that the person asking the question is an employee who is not familiar with tax law.
        
        Do not answer questions that are not related to United States federal tax laws.
        Politely decline to answer questions that are about tax policy, state taxes, or tax reform.
        If necessary, ask clarifying questions.
        
        ** Question **
        {question}
    """)

    output_parser = StrOutputParser()
    chain = prompt | model | output_parser
    response = chain.invoke({"question": question})
    print(question, '->', response)
    return response


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



def download_pdf(url: str, filename: str) -> str:
    """Downloads a PDF using urllib.request"""
    try:
        urllib.request.urlretrieve(url, filename)
        print(f"PDF downloaded successfully: {filename}")
    except Exception as e:
        print(f"An error occurred: {e}")
    return filename


def display_pdf(file):
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    # Embedding PDF in HTML
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)


def display_message(message: dict):
    if message["role"] == "Assistant":
        avatar = "🤖️"
    else:
        avatar = "🙋‍️"
    st.chat_message(message["role"]).write(message["content"])


if __name__ == '__main__':
    print("Gathering source documents")
    os.makedirs("cache", exist_ok=True)
    download_pdf("https://www.irs.gov/pub/irs-pdf/fw9.pdf", "cache/fw9.pdf")

    print("Initializing language model")
    model = VertexAI(model_name="gemini-pro") # add temperature if needed

    print("Initializing chat interface")
    st.title("W9 Copilot")
    st.caption("🙋‍♂️💬️🤖 Demo of an assistant for SaaS workflows")

    work_tab, history_tab = st.tabs(["Work", "History"])
    with work_tab:
        chat_pane, work_pane = st.columns([0.4, 0.6])
        with work_pane:
            st.header("Work pane")
            display_pdf("cache/fw9.pdf")

        with chat_pane:
            # clear chat history
            def clear_chat_history():
                st.session_state.messages = [{"role": "Assistant", "content": "How may I assist you today?"}]
            st.button('Clear Chat History', on_click=clear_chat_history)

            # display messages already in the session
            if "messages" not in st.session_state:
                st.session_state["messages"] = [
                    {"role": "Assistant",
                     "content": "️Hello! I'm your friendly copilot. How may I assist you today?"}
                ]
                for message in st.session_state["messages"]:
                    display_message(message)

            # input field for user to ask questions
            if prompt := st.chat_input("Ask question here..."):
                message = {"role": "User", "content": prompt}
                st.session_state.messages.append(message)
                display_message(message)

            # generate and display response from advisor
            if st.session_state.messages[-1]["role"] == "User":
                response = ask_question(model, st.session_state.messages)
                message = {"role": "Assistant", "content": response}
                st.session_state.messages.append(message)
                display_message(message)

    with history_tab:
        st.header("History")
        for message in st.session_state["messages"]:
            display_message(message)
