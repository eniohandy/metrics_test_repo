if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Avalia outputs RAG de um CSV.")
    parser.add_argument("--models-file", type=str, required=True)
    parser.add_argument("--data-file", type=str, required=True)
    args = parser.parse_args()

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
from imports import *

## Faithfulness está em ragas.metrics.collections e não em ragas.metrics
from ragas.metrics.collections import Faithfulness 

# nomes dos modelos a serem testados; casos de teste
# arquivos em formato json
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
    print ("\u2500" * 10, "Modelo",  "\u2500" * 10)
    print(f"----- {model_name} -----")
    print ("\u2500" * 28)
    llm= llm_factory(model_name, provider="openai", client=client)
    ### Create metric
    scorer = Faithfulness(llm=llm)
    ###
    print ()
    print ("\u2500" * 17, "Scorer", "\u2500" * 17)
    print (scorer)
    print ("\u2500" * 40)
    print ()
        
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
                response=case['reference'],
                retrieved_contexts=case['retrieved_contexts']
            )

            print ()
            print ("\u2500" * 15, "Pontuação", "\u2500" * 15)
            print(f"Context Precision Score: {result.value}")
            print ("\u2500" * 40)
            print ()