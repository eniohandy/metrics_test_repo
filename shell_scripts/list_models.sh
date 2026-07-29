### test the script to list LLM models available on the server

# curl http://$OLLAMA_SERVER:11434/api/tags | jq ".models[] | .name"  ### gera os nomes
curl http://$OLLAMA_SERVER:11434/api/tags | jq '[.models[] | {model: .model}]'

#-d '{  
#  "model": "nemotron-3-nano",  
#  "prompt": "quem é você?",  
#  "stream": false
#}' 