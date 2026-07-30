curl http://$OLLAMA_SERVER:11434/api/generate -d '{  
  "model": "laguna-xs-2.1:latest",  
  "prompt": "Why is the sky blue?",  
  "stream": false
}' | jq -r '.response'