curl http://$OLLAMA_SERVER:11434/api/generate -d '{  
  "model": "glm-4.7-flash",  
  "prompt": "Why is the sky blue?",  
  "stream": false
}' 
### check the raw answer without jq ###
### | jq -r '.response' ###