import json

def load_test_cases(data_file: str) -> list[dict]:
    with open(data_file) as f:
        return json.load(f)