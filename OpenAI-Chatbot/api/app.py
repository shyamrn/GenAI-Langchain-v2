# API v1.0

#* Import libraries
from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain_community.llms import Ollama
from langserve import add_routes
import uvicorn
import os
from dotenv import load_dotenv

#* Load environment variables
load_dotenv()

#* Declare app
app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="An API Server"
)

#* Add routes
add_routes(
    app,
    ChatOpenAI(),
    path="/openai"
)

#* Declare model
paid_llm = ChatOpenAI()

#*Declare Ollama model
open_llm = Ollama(model="mistral")

#*Declare prompts
paid_llm_prompt = ChatPromptTemplate.from_template("Write an essay on {topic} in less than 200 words.")
open_llm_prompt = ChatPromptTemplate.from_template("Write a poem on {topic} in less than 200 words.")

#* Add routes
add_routes(
    app,
    paid_llm_prompt|paid_llm,
    path="/essay"
)

add_routes(
    app,
    open_llm_prompt|open_llm,
    path="/poem"
)

#* Main
if __name__=="__main__":
    uvicorn.run(app, host="localhost", port=8000)