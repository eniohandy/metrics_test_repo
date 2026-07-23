# metrics_test_repo
This is a repo to test RAGAS metrics using Ollama and local LLMs

Ajustar as variáveis de ambiente para esconder o servidor.

Depois, há alguns scripts na pasta basic para teste fazendo chamadas curl
e outros scripts python, testanto
chat
generate
tags


Observações importantes:
no servidor local, estou usando WSL do windows e
venv do python. Tem que ativar para encontrar as bibliotecas.
instalei tb um pacote jq que faz parser de arquivos json.

Carreguei a bib ollama. Mas tem que fazer o teste com muitas outras.

Lista de modelos anteriores:

qwen3.6:27b
kimi-k2.7-code:cloud
zerocopia/cogito-2.1:671b-cloud
cogito:latest
vicuna:latest
glm-5.1:cloud
kimi-k2.6:cloud
deepseek-v4-flash:cloud
deepseek-v4-pro:cloud
minimax-m2.7:cloud
cogito-2.1:671b-cloud
qwen3-next:latest
qwen3-next:80b-cloud
gpt-oss:20b
gpt-oss:20b-cloud
kimi-k2.5:cloud
gpt-oss:120b-cloud
nemotron-3-nano:30b
glm-4.6:cloud
glm-4.7:cloud
llama3.1:latest
deepseek-r1:latest
gpt-oss:latest

|Lista de modelos atuais:

nemotron-3-nano:30b 
cogito:latest       
llama3.1:latest     
gpt-oss:latest      
deepseek-r1:latest  
llama3.1:70b        