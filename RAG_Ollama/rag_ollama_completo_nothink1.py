import numpy as np
import requests
import os

# ============================================================
# CONFIGURAÇÃO
# ============================================================

ollama_server = os.environ["OLLAMA_SERVER"]
OLLAMA_HOST = f"http://{ollama_server}:11434"
OLLAMA_EMBED_MODEL = "nomic-embed-text"
OLLAMA_CHAT_MODEL = "granite4.2:8b"


# ============================================================
# FUNÇÕES
# ============================================================

def get_embeddings(texts):
    if isinstance(texts, str):
        texts = [texts]
    response = requests.post(
        f"{OLLAMA_HOST}/api/embed",
        headers={"Content-Type": "application/json"},
        json={"model": OLLAMA_EMBED_MODEL, "input": texts},
    )
    return response.json()["embeddings"]


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def retrieve(query, document_embeddings, top_n=2):
    response = requests.post(
        f"{OLLAMA_HOST}/api/embed",
        headers={"Content-Type": "application/json"},
        json={"model": OLLAMA_EMBED_MODEL, "input": query},
    )
    query_embedding = np.array(response.json()["embeddings"][0])

    scored = []
    for doc in document_embeddings:
        score = cosine_similarity(query_embedding, np.array(doc["embedding"]))
        scored.append({"text": doc["text"], "score": float(score)})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_n]


def generate_answer(query, context_docs):
    context = "\n\n".join(
        f"[{i+1}] {doc['text']}" for i, doc in enumerate(context_docs)
    )

    response = requests.post(
        f"{OLLAMA_HOST}/api/chat",
        headers={"Content-Type": "application/json"},
        json={
            "model": OLLAMA_CHAT_MODEL,
            "think": False,
            "stream": False,
            "options": {"temperature": 0.2},
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

    return response.json()["message"]["content"]


# ============================================================
# EXECUÇÃO
# ============================================================

# PARTE I — Chunks de texto
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

# PARTE II/III — Indexar os chunks (gerar e guardar os embeddings)
embeddings = get_embeddings(chunks)
document_embeddings = [
    {"text": chunks[i], "embedding": embeddings[i]}
    for i in range(len(chunks))
]

print(f"Indexados {len(document_embeddings)} chunks, cada um com {len(document_embeddings[0]['embedding'])} dimensões")

# PARTE IV — Buscar os chunks mais relevantes pra pergunta
query = "quem tem direito à vale alimentação e refeição?"
results = retrieve(query, document_embeddings, top_n=2)

print("\nDocumentos recuperados:")
for i, r in enumerate(results):
    print(f"  {i+1}. (score: {r['score']:.4f}) {r['text']}")

# PARTE V — Gerar a resposta final
answer = generate_answer(query, results)
print(f"\nPergunta: {query}")
print(f"Resposta: {answer}")
