### test the script to list LLM models available on the server

curl http://$OLLAMA_SERVER:11434/api/tags | jq ".models[] | .name"

#-d '{  
#  "model": "nemotron-3-nano",  
#  "prompt": "quem é você?",  
#  "stream": false
#}' 