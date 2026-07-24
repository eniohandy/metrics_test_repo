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
# ocultar o servidor
ollama_server = os.environ['OLLAMA_SERVER']
host=f'http://{ollama_server}:11434/v1'
## Setup LLM - aqui será o ajuste para Ollama models
client = AsyncOpenAI(
    api_key="ollama",
    base_url=host
)
#
llm = llm_factory("gpt-oss", provider="openai", client=client)
print (llm.model)
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
input = "quanto gastei na padaria, lavanderia e restaurante??"
reference = "padaria R$ 100, lavanderia R$ 50,  restaurante R$ 120, cinema R$ 40"
retrieved = ["supermercado gastou R$250",
            "padaria gastou 100 reais",
            "lavanderia gastou R$ 50", 
            "restaurante gastou R$ 120.",
            "academia gastou R$ 200.",
            "loja gastou R$ 500.",
]

print (" ") 
print ("input: ", input)
print ("response: ", reference)
print ("retrieved contexts: ", retrieved)
print (" ")
#
result = scorer.score(
    user_input=input,
    response=reference,
    retrieved_contexts=retrieved
     
)
print(f"Faithfulness Score: {result.value}")