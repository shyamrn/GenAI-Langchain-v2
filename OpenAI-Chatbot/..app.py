# Chatbot v1.0

#* Import Libraries
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
import os
from dotenv import load_dotenv

#* Load environment variables
load_dotenv()

#* Define prompt template
prompt = ChatPromptTemplate.from_messages(
    
    [
        ("system", "You are a helpful assistant. Please respond to the user queries."),
        ("user", "Question: {question}")
    ]
)

#* Streamlit framework
st.title("Langchain chatbot using OpenAI")
input_text = st.text_input("Type your question here.")

#* OpenAI LLM
llm = ChatOpenAI(model="gpt-3.5-turbo")
output_parser = StrOutputParser()
chain = prompt|llm|output_parser

submit = st.button("Submit")
if submit:
    st.write(chain.invoke({"question":input_text}))
