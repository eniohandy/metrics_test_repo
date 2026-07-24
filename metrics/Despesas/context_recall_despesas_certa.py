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
from ragas.metrics.collections import ContextRecall
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
llm = llm_factory("nemotron-3-nano:30b", provider="openai", client=client)

### Create metric
scorer = ContextRecall(llm=llm)
###
print ("___ Scorer___")
print (scorer)
print ("___ Scorer___")
#
###
#### Evaluate
#### Neste caso, ContextRecall, são comparados os claims da "reference", separados pela LLM
#### com os claims contidos no "retrieved". O que conta é se cada claim reference aparece no retrieved.
#### caso afirmativo, a pontuação é total (1.0). Do contrário, uma fração.
#input = "o que é um cartão de crédito?"
#reference = "é um instrumento de crédito pessoal. existem nas bandeiras visa e master."
### no caso abaixo, todos os claims da reference estão no retrieved. Assim a resposta é 1.
#retrieved = ["os bancos emitem para seus clientes", 
#            "esta é só uma frase para ver se o resultado muda.",
#            "é um instrumento de crédito pessoal",
#            "existem nas bandeiras visa e master"
#]
### no caso abaixo, apenas 1 dos 2 claims da reference está no retrived. Assim, a resposta é 0,5
#retrieved = ["os bancos emitem para seus clientes", 
#            "esta é só uma frase para ver se o resultado muda.",
#            "é um instrumento de crédito pessoal",
#]
#input = "quanto gastei na padaria, lavanderia e restaurante??"
#reference = "padaria R$ 100, lavanderia R$ 50,  restaurante R$ 120"
#retrieved = ["padaria gastou 100 reais",
#            "lavanderia gastou R$ 50", 
#            "restaurante gastou R$ 120.",
#            "academia gastou R$ 200.",
#            "loja gastou R$ 500.",
#]
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