
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import os
from dotenv import load_dotenv
load_dotenv()
import json

queries = [
    "How can I import to use watsonx models?",
    "How can I use SearchApi as part of a Self Ask chain?",
    "How can I load a Wikipedia document?",
    "How can I use a Wolfram alpha Tool?",
    "Is ther any way to check the wheather?",
    "Is there a way to use decorators?",
    "Is the a way to use voice?",
    "How to use Serper - Google Search API as part of a Self Ask chain?",
    "What is Groq?",
    "How to use Llama.cpp embeddings?",
    "How to use GraphCypher?",
    "How to build a knowledge graph from text?",
    "What is Chroma?",
    "What is Alchemy?",
    "Is possible to use beautiful soap?"

]

def generate_response(queries, embeddings_provider):
    
    if embeddings_provider == 'HF':
        embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=os.getenv('HF_TOKEN'), model_name="sentence-transformers/all-MiniLM-l6-v2"
        )
        directory = "./hf_collection"
        print("HF db loaded")
        
    if embeddings_provider == 'Google':
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        directory = "./google_collection"
        print("Google db loaded")

    if embeddings_provider == 'OpenAI':
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        directory = "./langchain_collection"
        print("OpenAI db loaded")
    
    collection_name = f"langchain_collection_{embeddings_provider}_embeddings"
    print(collection_name)
    vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=directory
        )  

    
    results = [
        vector_store.similarity_search_by_vector_with_relevance_scores(
        embedding=embeddings.embed_query(query),
        k=3) for query in queries
    ]
    
    documents_dict = [
        [{"page_content": doc[0].page_content, "metadata": doc[0].metadata} for doc in sublist] for sublist in results
    ]

    # Save the list of dictionaries to a JSON file
    with open(f"results/query_{embeddings_provider}_results.json", "w") as file:
        json.dump(documents_dict, file, indent=4)

    print(f"Documents have been saved to 'query_{embeddings_provider}_results.json'")

print("starting HF validation:")
generate_response(queries, 'HF')
print("starting Google validation:")
generate_response(queries, 'Google')
print("starting OpenAI validation:")
generate_response(queries, 'OpenAI')