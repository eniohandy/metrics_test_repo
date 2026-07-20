curl http://200.144.192.87:11434/api/generate -d '{  
  "model": "llama3.1",  
  "prompt": "Why is the sky blue?",  
  "stream": false
}' | jq -r '.response'