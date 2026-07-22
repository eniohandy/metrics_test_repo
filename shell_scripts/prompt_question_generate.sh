curl http://$OLLAMA_SERVER:11434/api/generate -d '{  
  "model": "llama3.1",  
  "prompt": "Why is the sky blue?",  
  "stream": false
}' | jq -r '.response'