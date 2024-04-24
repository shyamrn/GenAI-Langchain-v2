# Langchain Open Source End-to-End Application - v1.0

#* Import libraries
from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import numpy as np
import os
from dotenv import load_dotenv

#* Load environment variables
load_dotenv()

#* Read the PDFs from Documents folder
loader = PyPDFDirectoryLoader("./Documents")
documents = loader.load()

#* Chunk the documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
final_documents = text_splitter.split_documents(documents)
#print(final_documents[0].page_content)
#print(len(final_documents))

#* Apply embeddings to chunk documents
huggingface_embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={'device':'cpu'},
    encode_kwargs={'normalize_embeddings':'True'}
)

np_array_test = np.array(huggingface_embeddings.embed_query(final_documents[0].page_content))
#print(np_array_test)
#print(np_array_test.shape)

#* Declare vector db
vector_db = FAISS.from_documents(final_documents[:120], huggingface_embeddings)
#print(vector_db)

#* Query using similarity search
input_text = "What is health insurance coverage?"
relevant_documents = vector_db.similarity_search(input_text)
#print(relevant_documents[0].page_content)

retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={'k':3})
#print(retriever)

#* Define prompt template
prompt_template_general = ChatPromptTemplate.from_messages(
    
    [
        ("system", "You are a helpful assistant. Please respond to the user queries."),
        ("user", "Question: {question}")
    ]
)

#* Declare LLM
huggingfacehub_llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-v0.1",
    model_kwargs={"temperature":0.1, "max_length":500}
)

llm = ChatOpenAI(model="gpt-3.5-turbo")
output_parser = StrOutputParser()
chain_general = prompt_template_general|llm|output_parser

response_general = chain_general.invoke({"question":input_text})
#print(response_general)

prompt_template="""
Use the following piece of context to answer the question asked.
Please try to provide the answer only based on the context

{context}
Question:{question}

Helpful Answers:
 """
 
prompt=PromptTemplate(template=prompt_template,input_variables=["context","question"])

retrievalQA=RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt":prompt}
)

query="""DIFFERENCES IN THE
UNINSURED RATE BY STATE
IN 2022"""

#* Call the QA chain with query
result = retrievalQA.invoke({"query": query})
print(result['result'])
