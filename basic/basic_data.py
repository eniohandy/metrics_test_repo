import os
import requests
import json

ollama_server = os.environ['OLLAMA_SERVER']

print (ollama_server)

# OLLAMA_URL = "http://{ollama_server}:11434/api/show"  
### response = requests.post(OLLAMA_URL, json={"model": "llama3.1:latest"})  
response = requests.post("http://{ollama_server}:11434/api/show", json={"model": "llama3.1:latest"})  
# print(response.json())
data = response.json()

# Mostrar dados formatados, estrutura completa (inclusive aninhada)
#print(json.dumps(data, indent=2, ensure_ascii=False))
print ("indices do modelo")
print(list(data.keys()))
#print(list(data.get("details", {}).keys()))
print ("***")
#print(data["model_info"])
for key, value in data["model_info"].items():
    print(f"{key}: {value}")

print(data["details"])
for key, value in data["details"].items():
    print(f"{key}: {value}")