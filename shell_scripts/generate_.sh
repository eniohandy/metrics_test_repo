curl -N http://200.144.192.67:3000/api/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-cbca990e6da14666ab7f2291d2fd1669" \
  -d '{
    "model": "llama3.1:latest",
    "messages": [
      {"role": "user", "content": "quem tem direito à vale alimentação e refeição?"}
    
    ],
    "files": [
      {"type": "collection", "id": "82cb8931-f8e4-4e54-8134-a46e15346521"}
    ]
  }'