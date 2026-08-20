curl -N http://200.144.192.87:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
  "model": "llama3.1",
  "messages": [
    {"role": "user", "content": "Hello"}
  ]
}'