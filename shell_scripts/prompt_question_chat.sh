curl http://$OLLAMA_SERVER:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "cogito",
    "messages": [{"role": "user", "content": "Why is the sky blue?"}],
    "stream": false
  }' | jq -r '.message.content'