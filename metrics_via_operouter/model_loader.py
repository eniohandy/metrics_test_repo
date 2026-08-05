import json

def load_model_names(models_file: str) -> list[str]:
    with open(models_file) as f:
        models_data = json.load(f)
    return [entry["model"] for entry in models_data]
    