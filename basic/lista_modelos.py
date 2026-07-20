### query direta ao ollama para obter os modelos disponiveis ###
### versão controlada pelo github ###
import requests
import json

OLLAMA_URL = "http://200.144.192.87:11434/api/tags"  # troque localhost pelo IP do servidor se rodar remotamente

response = requests.get(OLLAMA_URL)
response.raise_for_status()
data = response.json()

for model in data["models"]:
    print(model["name"])
                            