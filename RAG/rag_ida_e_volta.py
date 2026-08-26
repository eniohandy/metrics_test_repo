import numpy as np
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

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve(query, document_embeddings, top_n=2):
    response = requests.post(
        "https://openrouter.ai/api/v1/embeddings",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={"model": "openai/text-embedding-3-small", "input": query},
    )
    query_embedding = np.array(response.json()["data"][0]["embedding"])

    scored = []
    for doc in document_embeddings:
        score = cosine_similarity(query_embedding, np.array(doc["embedding"]))
        scored.append({"text": doc["text"], "score": float(score)})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_n]

query = "Quantos dias de licença tenho após 7 anos na empresa?"
results = retrieve(query, document_embeddings, top_n=2)

print("\nDocumentos recuperados:")
for i, r in enumerate(results):
    print(f"  {i+1}. (score: {r['score']:.4f}) {r['text']}")


def generate_answer(query, context_docs):
    context = "\n\n".join(
        f"[{i+1}] {doc['text']}" for i, doc in enumerate(context_docs)
    )

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": "openai/gpt-4o-mini",
            "messages": [
                {
                    "role": "system",
                    "content": "Responda a pergunta do usuário com base apenas no contexto fornecido. Cite o número da fonte entre colchetes.",
                },
                {
                    "role": "user",
                    "content": f"Contexto:\n{context}\n\nPergunta: {query}",
                },
            ],
        },
    )

    return response.json()["choices"][0]["message"]["content"]

answer = generate_answer(query, results)
print(f"\nPergunta: {query}")
print(f"Resposta: {answer}")