# RAG v1.0

#* Import Libraries
from langchain_community.document_loaders import TextLoader, WebBaseLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import bs4
import os
from dotenv import load_dotenv

#* Load environment variables
load_dotenv()

#* Declare text loader
text_loader = TextLoader("speech.txt")

text_text_documents = text_loader.load()
#print(text_text_documents)

#* Declare web base loader
web_base_loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(parse_only=bs4.SoupStrainer(
        class_=("post-title","post-content","post-header")
    ))
    )

web_text_documents = web_base_loader.load()
#print(web_text_documents)

#* Declare PDF loader
pdf_loader = PyPDFLoader("attention.pdf")
pdf_documents = pdf_loader.load()
#print(pdf_documents)

#* Declare text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

chunked_documents_pdf = text_splitter.split_documents(pdf_documents)
#print(chunked_documents_pdf)

#* Declare Chroma vector DB
chroma_db = Chroma.from_documents(chunked_documents_pdf[:30], OpenAIEmbeddings())
"""
#* Chroma query
query_1 = "Who are the authors of 'Attention is all you need' research paper?"
query_2 = "What is an attention function?"
chroma_result = chroma_db.similarity_search(query=query_2)
print("=======================================================================================================")
print("Result from Chroma DB")
print("=======================================================================================================")
print(chroma_result[0].page_content)
print("=======================================================================================================")
"""
#* Declare FAISS vector DB
faiss_db = FAISS.from_documents(chunked_documents_pdf[:30], OpenAIEmbeddings())
"""
#* FAISS query
query_3 = "What is an attention function?"
faiss_result = faiss_db.similarity_search(query=query_3)
print("=======================================================================================================")
print("Result from FAISS DB")
print("=======================================================================================================")
print(faiss_result[0].page_content)
print("=======================================================================================================")
print(faiss_db)
"""
#* Declare LLM
llm = Ollama(model="llama2")
prompt_template_1 = """
Answer the following question based only on the provided context.
Think step by step before providing a detailed answer.
<context>
{context}
<context>
Question: {input}
"""
prompt = ChatPromptTemplate.from_template(prompt_template_1)
document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
#print(document_chain)

faiss_retriever = faiss_db.as_retriever()
#print(faiss_retriever)

#* Define retrieval chain
faiss_retrieval_chain = create_retrieval_chain(faiss_retriever, document_chain)
faiss_question_1 = "What is an attention function?"
faiss_question_2 = "Describe the encoder and decoder used in this research."
faiss_retrieval_answer = faiss_retrieval_chain.invoke({"input":faiss_question_2})
print("=======================================================================================================")
print(faiss_retrieval_answer["answer"])
print("=======================================================================================================")
