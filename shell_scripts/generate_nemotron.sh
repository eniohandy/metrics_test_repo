curl http://$OLLAMA_SERVER:11434/api/generate -d '{  
  "model": "nemotron-3-nano",  
  "prompt": "quem é você?",  
  "stream": false
}' ### | jq -r '.response' 
### check the raw answer without jq ###