import numpy as np
import requests
import os

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "COLE_SUA_KEY_AQUI")

chunks = [
"Abaixo está a lista de benefícios oferecidos aos colaboradores da Riachuelo, conforme as informações oficiais do portal de carreiras da empresa:",
"1. Refeição Descrição: Os colaboradores contam com alimentação de qualidade em restaurantes internos. Para quem está em trabalho remoto, é oferecido um auxílio que pode ser utilizado em restaurantes e supermercados. Nas lojas, todos recebem vale-alimentação ou vale-refeição.",
"2. Auxílio trabalho remoto e VT (Vale Transporte) Descrição: Colaboradores em regime de trabalho remoto recebem um auxílio mensal para apoiar nas despesas de energia elétrica e internet. Adicionalmente, é oferecido vale-transporte para todos os colaboradores, de acordo com o desejo de cada um.",
"3. Previdência Privada Descrição: Plano de Previdência Privada com o intuito de apoiar os colaboradores no investimento mensal pensando no longo prazo e no futuro em todas as etapas da vida.",
"4. Cartão Flex Descrição: Um cartão para apoiar as despesas do mês, que dá acesso a estabelecimentos como farmácias, postos de combustíveis e supermercados, com o valor gasto sendo descontado na folha de pagamento.",
"5. Desconto na Riachuelo Descrição: Descontos especiais em compras realizadas nas lojas físicas, site e aplicativo utilizando o Cartão Riachuelo. Os descontos podem ser ainda maiores dependendo da campanha do mês, como Natal, Dia das Mães e Black Friday.",
"6. Assistência Médica e Odontológica Descrição: Planos abrangentes de assistência médica e odontológica com rede credenciada de qualidade em todo o país para cuidar da saúde do colaborador e de seus familiares.",
"7. Foco em saúde e bem-estar Descrição: Programas de saúde e qualidade de vida que incluem campanhas de vacinação, acompanhamento de doenças crônicas, programas para gestantes, psicólogos online, descontos em medicamentos, entre outros.",
"8. Desenvolvimento e cultura Descrição: Parcerias com instituições de ensino e escolas de idiomas que oferecem descontos especiais em cursos de graduação, especialização, MBA e idiomas, além de descontos em shows, exposições e teatros.",
"9. Gympass Descrição: Parceria que oferece acesso ilimitado às melhores academias, estúdios, aulas, treinos e aplicativos de bem-estar em um único benefício para incentivar os cuidados com a saúde física e mental.",
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

query = "quem tem direito à vale alimentação e refeição?"
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
        "model": "google/gemini-3.7-flash",
        "reasoning": {"effort": "medium"},
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