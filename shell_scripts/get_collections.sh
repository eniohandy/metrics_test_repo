curl -s http://$OLLAMA_SERVER:3000/api/v1/knowledge/ \
  -H "Authorization: Bearer sk-f8d0da7fe28d4477bd8bef2ec4ae5db2" \
  | jq -r '.[] '