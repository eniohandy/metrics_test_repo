#import argparse
#import json

#parser = argparse.ArgumentParser()
#parser.add_argument("--models-file", type=str, required=True, help="Caminho pro JSON com a lista de modelos")
#args = parser.parse_args()

#with open(args.models_file) as f:
#    models_data = json.load(f)              

#model_names = [entry["model"] for entry in models_data]   

#for model_name in model_names:              
#   print(f"Rodando avaliação para: {model_name}")


import json

def load_model_names(models_file: str) -> list[str]:
    with open(models_file) as f:
        models_data = json.load(f)
    return [entry["model"] for entry in models_data]
    