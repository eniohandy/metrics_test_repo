### este script está feito com o modulo Client do Ollama

import os
from ollama import Client 
host=os.environ['OLLAMA_SERVER']

client = Client(host)

#response = client.generate(model='cogito', prompt="quanto gastei  em a, b e c? a = 10, b = 20 e c = 30. Informe os valores para cada um")
# response = client.generate(model='cogito', prompt="gastei 10 na padaria, 20 no restaurante e R$ 30 na lavanderia. Minhas referências dizem R$10 padaria,\
# 20 restaurante e R$3 lavanderia. Onde está o erro?")

prompt="quantos itens tem na lista 1 e na lista 2? qual a diferença entre a lista1 e a lista2? responda apenas apontando os erros. lista1:supermercado gastou R$250, \
padaria gastou 100 reais, lavanderia gastou R$ 50, restaurante gastou R$ 120, \
academia gastou R$ 200, loja gastou R$ 500, \
lista2: supermercado gastou R$250, padaria gastou 10 reais, lavanderia gastou R$ 50, restaurante gastou R$ 120, academia gastou R$ 200, loja gastou R$ 500"

response = client.generate(model='gpt-oss', prompt=prompt)

resp = response.model_dump_json()

print(resp)