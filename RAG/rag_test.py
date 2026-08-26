import requests
import os

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "COLE_SUA_KEY_AQUI")

chunks = [
    "Artigo 12: Colaboradores de Engenharia de Dados têm direito a 3 dias de recesso criativo por trimestre.",
    "Artigo 18: O código de aprovação para reembolso de home office acima de R$ 800 é GAMMA-9.",
    "Artigo 25: Colaboradores com 7 anos de empresa recebem a Licença Fenda, de 12 dias corridos.",
]

response = requests.post(
    "https://openrouter.ai/api/v1/embeddings",
    headers={
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    },
    json={"model": "openai/text-embedding-3-small", "input": chunks},
)

data = response.json()
document_embeddings = [
    {"text": chunks[item["index"]], "embedding": item["embedding"]}
    for item in data["data"]
]

print(f"Indexados {len(document_embeddings)} chunks, cada um com {len(document_embeddings[0]['embedding'])} dimensões")