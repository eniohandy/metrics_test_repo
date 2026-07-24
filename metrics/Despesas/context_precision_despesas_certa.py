import os
import sys
from pathlib import Path
import asyncio
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

## bibliotecas Ollama, OpenAI e Ragas
from ollama import Client

from openai import AsyncOpenAI

from ragas.llms import llm_factory

## ContextPrecision está em ragas.metrics.collections e não em ragas.metrics
from ragas.metrics.collections import ContextPrecision
#
# ocultar o servidor
ollama_server = os.environ['OLLAMA_SERVER']
host=f'http://{ollama_server}:11434/v1'
#
## Setup LLM - aqui será o ajuste para Ollama models
client = AsyncOpenAI(
    api_key="ollama",
    base_url=host
)
#
# llm = llm_factory("nemotron-3-nano:30b", provider="openai", client=client)
llm = llm_factory("cogito", provider="openai", client=client)
print ("modelo: ", llm.model)

### Create metric
scorer = ContextPrecision(llm=llm)
###
print ("___ Scorer___")
print (scorer)
print ("___ Scorer___")
#
###
#### Evaluate
#### Neste caso, o que é comparado são as tres frases "retrieved" contra a "reference" 

input = "quanto gastei na padaria, lavanderia e restaurante??"
reference = "padaria foram 100 reais, lavanderia R$ 50 e restaurante, R$ 120"
retrieved = ["padaria gastou 100 reais, lavanderia gastou R$ 50, restaurante gastou R$ 120.",
]

print (" ") 
print ("input: ", input)
print ("reference: ", reference)
print ("retrieved contexts: ", retrieved)
print (" ")
#
result = scorer.score(
    user_input=input,
    reference= reference,
    retrieved_contexts=retrieved
     
)
print(f"Context Precision Score: {result.value}")