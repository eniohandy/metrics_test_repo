curl -N http://200.144.192.87:3000/api/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-f8d0da7fe28d4477bd8bef2ec4ae5db2" \
  -d '{
    "model": "qwen3:latest",
    "messages": [
      {"role": "user", "content": "quantos dias de férias eu tenho direito?"}
    
    ],
    "files": [
      {"type": "collection", "id": "333f534f-fe6a-457b-b5de-86cf58891ef0"}
    ]
  }'