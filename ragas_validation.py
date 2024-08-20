
from ragas import evaluate
from datasets import Dataset
from ragas.metrics import context_precision, context_recall, context_entity_recall
import json
from dotenv import load_dotenv
from langchain.schema import Document
load_dotenv()

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

ground_truths = [
  """Install the integration package with pip install -qU langchain-ibm. Then get an IBM watsonx.ai api key and set it as an environment variable (`WATSONX_APIKEY`)
  import os      
  os.environ["WATSONX_APIKEY"] = "your IBM watsonx.ai api key"  
  Chat Model: from langchain_ibm import ChatWatsonx
  """,
  """from langchain_community.utilities import SearchApiAPIWrapper
from langchain_openai import OpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
import os
os.environ["SEARCHAPI_API_KEY"] = ""
os.environ['OPENAI_API_KEY'] = ""
llm = OpenAI(temperature=0)
search = SearchApiAPIWrapper()
tools = [
    Tool(
        name="Intermediate Answer",
        func=search.run,
        description="useful for when you need to ask with search"
    )
]
self_ask_with_search = initialize_agent(tools, llm, agent=AgentType.SELF_ASK_WITH_SEARCH, verbose=True)
self_ask_with_search.run("Who lived longer: Plato, Socrates, or Aristotle?")""",
"""Document Loader
from langchain_community.document_loaders import WikipediaLoader""",
"""Tool
You can also easily load this wrapper as a Tool (to use with an Agent). You can do this with:

from langchain.agents import load_tools
tools = load_tools(["wolfram-alpha"])""",
"""OpenWeatherMap
OpenWeatherMap provides all essential weather data for a specific location:

Current weather
Minute forecast for 1 hour
Hourly forecast for 48 hours
Daily forecast for 8 days
National weather alerts
Historical weather data for 40+ years back""",
"""LangChain decorators is a layer on the top of LangChain that provides syntactic sugar 🍭 for writing custom langchain prompts and chains""",
"""ElevenLabs is a voice AI research & deployment company with a mission to make content universally accessible in any language & voice.
ElevenLabs creates the most realistic, versatile and contextually-aware AI audio, providing the ability to generate speech in hundreds of new and existing voices in 29 languages.""",
"""from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_openai import OpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType

import os

os.environ["SERPER_API_KEY"] = ""
os.environ['OPENAI_API_KEY'] = ""

llm = OpenAI(temperature=0)
search = GoogleSerperAPIWrapper()
tools = [
    Tool(
        name="Intermediate Answer",
        func=search.run,
        description="useful for when you need to ask with search"
    )
]

self_ask_with_search = initialize_agent(tools, llm, agent=AgentType.SELF_ASK_WITH_SEARCH, verbose=True)
self_ask_with_search.run("What is the hometown of the reigning men's U.S. Open champion?")""",
"""Welcome to Groq! 🚀 At Groq, we've developed the world's first Language Processing Unit™, or LPU. The Groq LPU has a deterministic, single core streaming architecture that sets the standard for GenAI inference speed with predictable and repeatable performance for any given workload.

Beyond the architecture, our software is designed to empower developers like you with the tools you need to create innovative, powerful AI applications. With Groq as your engine, you can:

Achieve uncompromised low latency and performance for real-time AI and HPC inferences 🔥
Know the exact performance and compute time for any given workload 🔮
Take advantage of our cutting-edge technology to stay ahead of the competition 💪""",
"""Embeddings
There exists a LlamaCpp Embeddings wrapper, which you can access with

from langchain_community.embeddings import LlamaCppEmbeddings""",
"""There exists a wrapper around Neo4j graph database that allows you to generate Cypher statements based on the user input and use them to retrieve relevant information from the database.

from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain""",
"""Constructing a knowledge graph from text
Text data often contain rich relationships and insights that can be useful for various analytics, recommendation engines, or knowledge management applications. Diffbot's NLP API allows for the extraction of entities, relationships, and semantic meaning from unstructured text data. By coupling Diffbot's NLP API with Neo4j, a graph database, you can create powerful, dynamic graph structures based on the information extracted from text. These graph structures are fully queryable and can be integrated into various applications.

from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers.diffbot import DiffbotGraphTransformer""",
"""Chroma is a database for building AI applications with embeddings.""",
"""Alchemy is the platform to build blockchain applications.""",
"""Installation and Setup
pip install beautifulsoup4
Document Transformer
from langchain_community.document_loaders import BeautifulSoupTransformer"""
]
def validation_embeddings(queries, ground_truths, embeddings_provider):
    
    # Load the list of dictionaries from the JSON file
    with open(f"results/query_{embeddings_provider}_results.json", "r") as file:
        documents_dict = json.load(file)

    # Convert the list of dictionaries back to a list of Document objects
    documents = [
        [Document(page_content=result["page_content"], metadata=result["metadata"])for result in doc]
        for doc in documents_dict
    ]
    results_page_contents = [[doc.page_content for doc in sublist] for sublist in documents]

    d = {
        "question": queries,
        #"answer": results,
        "contexts": results_page_contents,
        "ground_truth": ground_truths
    }

    dataset = Dataset.from_dict(d)
    #score = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall, context_entity_recall, answer_similarity, answer_correctness, harmfulness])
    score = evaluate(dataset, metrics=[context_precision, context_recall, context_entity_recall])
    print(score)
    score_df = score.to_pandas()
    score_df.to_parquet(f'./results/results_{embeddings_provider}_embeddings_split.parquet')
    print(f'results/results_{embeddings_provider}_embeddings_split.parquet generated')

print("starting HF validation:")
validation_embeddings(queries, ground_truths, 'HF')
print("starting Google validation:")
validation_embeddings(queries, ground_truths, 'Google')
print("starting OpenAI validation:")
validation_embeddings(queries, ground_truths, 'OpenAI')