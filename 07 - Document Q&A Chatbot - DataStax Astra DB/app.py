# Document Q&A Chatbot | DataStax Astra DB

#* Import libraries
from langchain_astradb import AstraDBVectorStore
from langchain.embeddings import OpenAIEmbeddings
#from langchain.vectorstores.cassandra import Cassandra
from datasets import load_dataset
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
import os
from dotenv import load_dotenv

#* Load environment variables
load_dotenv()

#* Embedding model and vector store
embedding = OpenAIEmbeddings()

vector_store = AstraDBVectorStore(
    collection_name=os.getenv('ASTRA_DB_COLLECTION_NAME'),
    embedding=embedding,
    token=os.getenv('ASTRA_DB_APPLICATION_TOKEN'),
    api_endpoint=os.getenv('ASTRA_DB_API_ENDPOINT')
)

print("=========================================================")
print("Astra vector store configuration completed")
print("---------------------------------------------------------")
print("Vector store: ", vector_store)
print("=========================================================")

#* Load dataset
philosophy_dataset = load_dataset("datastax/philosopher-quotes")['train']
#print("An example entry: ")
#print(philosophy_dataset[16])

#* Constructs a set of documents from your data. Documents can be used as inputs to your vector store.
docs = []
for entry in philosophy_dataset:
    metadata = {"author": entry["author"]}
    if entry["tags"]:
        #* Add metadata tags to the metadata dictionary
        for tag in entry["tags"].split(";"):
            metadata[tag] = "y"
    #* Create a LangChain document with the quote and metadata tags
    doc = Document(page_content=entry["quote"], metadata=metadata)
    docs.append(doc)

#print("=========================================================")
#print(docs)
#print("=========================================================")

# Create embeddings by inserting your documents into the vector store.
inserted_ids = vector_store.add_documents(docs)
print(f"\nInserted {len(inserted_ids)} documents.")

# Checks your collection to verify the documents are embedded.
#print(vector_store.astra_db.collection(os.getenv('ASTRA_DB_COLLECTION_NAME')).find())

#* Retrieval
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

prompt_template = """
Answer the question based only on the supplied context. If you don't know the answer, say you don't know the answer.
Context: {context}
Question: {question}
Your answer:
"""
prompt = ChatPromptTemplate.from_template(prompt_template)
model = ChatOpenAI()

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

answer = chain.invoke("In the given context, what is the most important to allow the brain and provide me the tags?")
print("Answer: ", answer)
