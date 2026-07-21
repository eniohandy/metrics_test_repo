import os
import requests
import json

ollama_server = os.environ['OLLAMA_SERVER']

print ("Verifica se a variavel do servidor está configurada.")
if ollama_server:
    print (ollama_server)
else:
    print ("não configurada")