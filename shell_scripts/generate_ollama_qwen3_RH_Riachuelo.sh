curl -N http://200.144.192.87:3000/api/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-f8d0da7fe28d4477bd8bef2ec4ae5db2" \
  -d '{
    "model": "qwen3:latest",
    "messages": [
      {"role": "user", "content": "quem tem direito à vale alimentação e refeição?"}
    
    ],
    "files": [
      {"type": "collection", "id": "f53c735f-1dbc-403d-9ae3-e30ca3ddcac2"}
    ]
  }'