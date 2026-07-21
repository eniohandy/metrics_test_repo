### query direta ao ollama para obter os modelos disponiveis ###
import os
import requests
import json

ollama_server = os.environ['OLLAMA_SERVER']
host=f'http://{ollama_server}:11434/api/tags'

response = requests.get(host)
response.raise_for_status()
data = response.json()

for model in data["models"]:
    print(model["name"])
                            