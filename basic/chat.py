from ollama import Client 

client = Client(host='http://200.144.192.87:11434')


messages = [
  {
    'role': 'user',
    'content': 'qual o dia mais longo do ano?',
  },
]

response = client.chat('gpt-oss', messages=messages)
print(response['message']['content'])