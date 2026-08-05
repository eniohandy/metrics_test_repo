if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Avalia outputs RAG de um CSV.")
    parser.add_argument("--models-file", type=str, required=True)
    parser.add_argument("--data-file", type=str, required=True)
    args = parser.parse_args()

#    parser.add_argument("--input",  required=True, help="Caminho para o CSV de entrada")
#   parser.add_argument("--output", required=True, help="Caminho para o CSV de saída")
#    args = parser.parse_args()


import os
import sys
import json
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

# para ler arquivos externos
import argparse
# minha função de importacao dos modelos
from model_loader import load_model_names
from data_loader import load_test_cases

### buscando os modelos a serem testados
#parser = argparse.ArgumentParser()
#parser.add_argument("--models-file", type=str, required=True)
#parser.add_argument("--data-file", type=str, required=True)
#args = parser.parse_args()

model_names = load_model_names(args.models_file)
test_cases = load_test_cases(args.data_file)

# ocultar o servidor
ollama_server = os.environ['OLLAMA_SERVER']
host=f'http://{ollama_server}:11434/v1'

## Setup LLM - aqui será o ajuste para Ollama models
client = AsyncOpenAI(
    api_key="ollama",
    base_url=host
)

#
for model_name in model_names:
    print ("___ Modelo ___")
    print(f"=== {model_name} ===")
    llm= llm_factory(model_name, provider="openai", client=client)
    ### Create metric
    scorer = ContextPrecision(llm=llm)
    ###
    print ("___ Scorer___")
    print (scorer)
    print ("______")
        
    for case in test_cases:
            user_input=case["user_input"],
            reference=case["reference"],
            retrieved_contexts=case["retrieved_contexts"],
            # print(case['user_input']) ### só pra ver se estava funcionando
            # print(user_input) ### idem
            #print ()
            #print(case['user_input'])
            #print(case['reference'])
            #print(case['retrieved_contexts'])

            result = scorer.score(
                user_input=case['user_input'],
                reference=case['reference'],
                retrieved_contexts=case['retrieved_contexts']
            )

            print ("___ Pontuação___")
            print(f"Context Recall Score: {result.value}")
            print ("______")