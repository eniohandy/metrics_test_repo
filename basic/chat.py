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

# response = client.chat('gpt-oss', messages=messages)
response = client.chat('cogito', messages=messages,
                      options={
                         'temperature':0,
                         'seed': 10,
                      }
)
print(response['message']['content'])