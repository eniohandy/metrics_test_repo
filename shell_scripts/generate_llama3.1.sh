curl -N http://$OLLAMA_SERVER2:3000/api/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $WEBUI_API_KEY" \
  -d '{
    "model": "llama3.1:latest",
    "messages": [
      {"role": "user", "content": "quem tem direito à vale alimentação e refeição?"}
    
    ],
    "files": [
      {"type": "collection", "id": "82cb8931-f8e4-4e54-8134-a46e15346521"}
    ]
  }'