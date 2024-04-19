#Client v1.0

#* Import libraries
import requests
import streamlit as st

def get_paid_llm_response(input_text):
    response = requests.post(
        "http://localhost:8000/essay/invoke",
        json={'input':{'topic':input_text}}
        )
    return response.json()['output']['content']

def get_open_llm_response(input_text):
    response = requests.post(
        "http://localhost:8000/poem/invoke",
        json={'input':{'topic':input_text}}
        )
    return response.json()['output']

#* Streamlit framework
st.title('Langserve with OpenAI and Ollama')
paid_llm_input_text = st.text_input("Write an essay on")
open_llm_input_text = st.text_input("Write a poem on")
submit_button = st.button("Submit")

if paid_llm_input_text and submit_button:
    st.write(get_paid_llm_response(paid_llm_input_text))
    
if open_llm_input_text and submit_button:
    st.write(get_open_llm_response(open_llm_input_text))
