curl http://$OLLAMA_SERVER:11434/api/embed \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nomic-embed-text",
    "input": ["teste de embedding de RH"]
  }'