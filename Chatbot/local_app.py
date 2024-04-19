# Chatbot v2.0

#* Import Libraries
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
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
st.title("Langchain chatbot using Ollama")
input_text = st.text_input("Type your question here.")

#* Ollama LLM
llm = Ollama(model="mistral")
output_parser = StrOutputParser()
chain = prompt|llm|output_parser

submit = st.button("Submit")
if submit:
    st.write(chain.invoke({"question":input_text}))