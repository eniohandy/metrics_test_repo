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

# para ler arquivos externos
import argparse
# minha função de importacao dos modelos
from model_loader import load_model_names
### do outro teste. Deixar desabilitado
# from ragas.embeddings import embedding_factory ## só precisa para AnswerCorrectness
# from ragas import evaluate
# from ragas.metrics import ContextPrecision, ContextRecall, Faithfulness, AnswerCorrectness
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
### buscando os modelos a serem testados
parser = argparse.ArgumentParser()
parser.add_argument("--models-file", type=str, required=True)
args = parser.parse_args()

model_names = load_model_names(args.models_file)
#
# llm = llm_factory("nemotron-3-nano:30b", provider="openai", client=client)
for model_name in model_names:
    print(f"=== Avaliando modelo: {model_name} ===")
    llm= llm_factory(model_name, provider="openai", client=client)

    ### Create metric
    scorer = ContextRecall(llm=llm)
    ###
    print ("___ Scorer___")
    print (scorer)
    print ("___ Scorer___")
    #

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