# Langchain Agents - v1.0

#* Import libraries
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_community.utilities import ArxivAPIWrapper
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain import hub
import os
from dotenv import load_dotenv

#* Load environment variables
load_dotenv()

#* Declare api wrapper
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

#print(wiki_tool)
#print(wiki_tool.name)

#* Declare loader
web_base_loader = WebBaseLoader("https://docs.smith.langchain.com/")
wiki_docs = web_base_loader.load()
wiki_documents = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(wiki_docs)

#* Declare vector db
vector_db = FAISS.from_documents(documents=wiki_documents, embedding=OpenAIEmbeddings())
wiki_retriever = vector_db.as_retriever()

#print(retriever)

langsmith_retriever_tool = create_retriever_tool(
    wiki_retriever, 
    "langsmith_search", 
    "Search for any information about LangSmith. You should use this tool for any questions related to LangSmith"
    )

arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)

#print(arxiv_tool)
#print(arxiv_tool.name)

#* Declare tools
tools = [langsmith_retriever_tool, wiki_tool]

#print(tools)

#* Declare LLM
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0.0)

#* Declare prompt | get prompt from LangChain Hub
prompt = hub.pull("hwchase17/openai-functions-agent")

#print(prompt.messages)

#* Declare agent and agent executor
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

#print(agent_executor)

#* Run the agent
result_1 = agent_executor.invoke({"input":"Tell me about LangSmith"})
result_2 = agent_executor.invoke({"input":"Tell me about Machine Learning"})
print("========================================================================================")
print(result_2)
print("========================================================================================")
