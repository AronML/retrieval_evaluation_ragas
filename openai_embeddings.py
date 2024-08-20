import tqdm as notebook_tqdm
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from uuid import uuid4
import os
import json
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

# Load the list of dictionaries from the JSON file
with open("input/documents_split_langchain.json", "r") as file:
    documents_dict = json.load(file)

# Convert the list of dictionaries back to a list of Document objects
documents = [
    Document(page_content=doc["page_content"], metadata=doc["metadata"])
    for doc in documents_dict
]

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
)

vector_store = Chroma(
    collection_name="langchain_collection_OpenAI_embeddings",
    embedding_function=embeddings,
    persist_directory="./langchain_collection", 
)

uuids = [str(uuid4()) for _ in range(len(documents))]

for i in range(0,len(documents),1000):
    if (len(documents)-i)<1000:
        vector_store.add_documents(documents=documents[i:], ids=uuids[i:])
    else:
        vector_store.add_documents(documents=documents[i:i+1000], ids=uuids[i:i+1000])



