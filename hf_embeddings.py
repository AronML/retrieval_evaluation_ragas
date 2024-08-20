from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from uuid import uuid4
import os
import json
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

embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=os.getenv('HF_TOKEN'), model_name="sentence-transformers/all-MiniLM-l6-v2"
)

vector_store = Chroma(
    collection_name="langchain_collection_HF_embeddings",
    embedding_function=embeddings,
    persist_directory="./hf_collection", 
)

uuids = [str(uuid4()) for _ in range(len(documents))]

for i in range(0,len(documents),1000):
    print(i)
    if (len(documents)-i)<1000:
        vector_store.add_documents(documents=documents[i:], ids=uuids[i:])
    else:
        vector_store.add_documents(documents=documents[i:i+1000], ids=uuids[i:i+1000])


