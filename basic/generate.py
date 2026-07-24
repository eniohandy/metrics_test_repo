### este script está feito com o modulo Client do Ollama
### comparado com outros python, é praticamente igual.


import os
from ollama import Client 
host=os.environ['OLLAMA_SERVER']

#client = Client(host='http://200.144.192.87:11434')
#client = Client(host='http://$OLLAMA_SERVER:11434')
client = Client(host)

messages = [
  {
    'role': 'user',
    'content': 'qual o dia mais longo do ano?',
  },
]

#response = client.generate('cogito', messages=messages,
#                      options={
#                         'temperature':0,
#                         'seed': 10,
#                      }
#)
response = client.generate(model='cogito', prompt="qual o dia mais curto do ano?")

print(response)