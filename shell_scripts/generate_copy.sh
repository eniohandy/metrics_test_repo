curl -N http://200.144.192.67:3000/api/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-cbca990e6da14666ab7f2291d2fd1669" \
  -d '{
    "model": "gemma4:latest",
    "messages": [
      {"role": "user", "content": "quais são deveres e direitos com relação à propriedade intelectual?"}
    
    ],
    "files": [
      {"type": "collection", "id": "c208c189-dec4-4bac-9009-3ce5cf13d52e"}
    ]
  }'