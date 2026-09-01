curl -N http://$OLLAMA_SERVER:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
  "model": "llama3.1",
  "messages": [
    {"role": "user", "content": "quantos dias de férias eu tenho direito?"}
  ],
  "temperature": 0.0,
  "seed": 17
}'