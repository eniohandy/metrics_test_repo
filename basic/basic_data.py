import os
import requests
import json

ollama_server = os.environ['OLLAMA_SERVER']

print (ollama_server)
host=f'http://{ollama_server}:11434/api/show'

response = requests.post(host, json={"model": "llama3.1:latest"})  
data = response.json()

# Mostrar dados formatados, estrutura completa (inclusive aninhada)
#print(json.dumps(data, indent=2, ensure_ascii=False))

print ("*** INDICES DO MODELO ***")
print(list(data.keys()))

print ("***")

print ("*** MODEL INFO ***")
for key, value in data["model_info"].items():
    print(f"{key}: {value}")

print ("*** MODEL DETAILS ***")
print(data["details"])
for key, value in data["details"].items():
    print(f"{key}: {value}")