# Document Q&A Chatbot - v1.0

#* Import libraries
import streamlit as st
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
import time
import os
from dotenv import load_dotenv

#* Load environment variables
load_dotenv()

#* Function: Document embedding
def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OpenAIEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("./Documents")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

#* Main application
st.title("Document Q&A Chatbot | Llama3 | Groq")

#* Declare LLM model
llm = ChatGroq(model_name = 'Llama3-8b-8192')

prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on provided context only. Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Question: {input}
    """
)



question = st.text_input("Ask your question from the documents")

if st.button("Documents Embedding"):
    vector_embedding()
    st.write("Vector store DB is ready!!!")

submit = st.button("Submit")

if question and submit:
    document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    start = time.process_time()
    response = retrieval_chain.invoke({'input':question})
    st.write("Response time: ", time.process_time()-start)
    st.write(response['answer'])
    
     # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write("=====================================")
