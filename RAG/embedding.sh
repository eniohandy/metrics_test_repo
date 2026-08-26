curl https://openrouter.ai/api/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENROUTER_API_KEY" \
  -d '{
    "model": "openai/text-embedding-3-small",
    "input": ["teste de embedding de RH"]
  }'