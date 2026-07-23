import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

host=os.environ['OLLAMA_SERVER']
# from openai import OpenAI
from openai import AsyncOpenAI
from datasets import Dataset
from ragas import evaluate
from ragas.llms import llm_factory
from ragas.metrics import (
    ContextPrecision,
    ContextRecall,
    Faithfulness,
    AnswerCorrectness,
)
### nova library do AnswerCorrectness ###
### precisa de embeddings ###
from ragas.embeddings import embedding_factory


# Initialize LLM with your provider
### import google.generativeai as genai  ### comentado do exemplo
### genai.configure(api_key="...") ### aqui basta usar ollama
### client = genai.GenerativeModel("gemini-2.0-flash")
# ocultar o servidor
ollama_server = os.environ['OLLAMA_SERVER']
host=f'http://{ollama_server}:11434/v1'

client = AsyncOpenAI(
    api_key="ollama",
    base_url=host
)

embeddings = embedding_factory(
    provider="openai",
    model="nomic-embed-text",   # troque pelo modelo de embeddings que você tem no Ollama
    client=client          # o mesmo client OpenAI apontando pro Ollama
)

### llm = llm_factory("llama3.1", provider="openai", client=client)
### llm = llm_factory("nemotron-3-nano:30b", provider="openai", client=client)
llm = llm_factory("deepseek-r1", provider="openai", client=client)

# Create evaluation dataset
#data = {
#    "question": ["What is the capital of France?"],
#    "answer": ["Paris"],
#    "contexts": [["France is in Europe. Paris is its capital."]],
#    "ground_truth": ["Nice"]
#}
data = {
    "question": ["Which are the two more important cities of France?"],
    "answer": ["Paris and Nice"],
    "contexts": [["France is in Europe. Paris is its capital. Other importante cities are Nice, Tolouse and Cannes."]],
    "ground_truth": ["Nice and Paris"]
}
dataset = Dataset.from_dict(data)

# Define metrics
metrics = [
    ContextPrecision(llm=llm),
    ContextRecall(llm=llm),
    Faithfulness(llm=llm),
    AnswerCorrectness(llm=llm, embeddings=embeddings), ### comentado para teste
]

# Evaluate
results = evaluate(dataset, metrics=metrics, raise_exceptions=True)
print(results)