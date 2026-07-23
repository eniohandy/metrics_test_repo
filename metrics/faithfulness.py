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
from ragas.metrics.collections import Faithfulness 
#
## Setup LLM - aqui será o ajuste para Ollama models
client = AsyncOpenAI(
    api_key="ollama",
    base_url="http://200.144.192.87:11434/v1"
)
#
llm = llm_factory("nemotron-3-nano:30b", provider="openai", client=client)

### Create metric
scorer = Faithfulness(llm=llm)
###
print ("___ Scorer___")
print (scorer)
print ("___ Scorer___")
#
###
#### Evaluate
#### Neste caso, o que é comparado são as tres frases "retrieved" contra a "reference" 
#### A ordem importa. Então, como a terceira frase é igual a reference, o score é 0,333.
input = "o que é um cartão de crédito?"
response = "é um instrumento de crédito pessoal"
# retrieved = ["cartão de crédito é um instrumento de crédito pessoal",] 
retrieved = ["os bancos emitem para seus clientes", 
           "esta é só uma frase para ver se o resultado muda.",
           "cartões de crédito são um instrumento de crédito pessoal",
]

print (" ") 
print ("input: ", input)
print ("response: ", response)
print ("retrieved contexts: ", retrieved)
print (" ")
#
result = scorer.score(
    user_input=input,
    response=response,
    retrieved_contexts=retrieved
     
)
print(f"Faithfulness Score: {result.value}")