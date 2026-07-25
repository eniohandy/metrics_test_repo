### Diference entre **Context Recall** e **Faithfulness** no RAGAS  

O framework **RAGAS** (Retrieval‑Augmented Generation Assessment) traz um conjunto de métricas que permitem avaliar, de forma 
automática, a qualidade das respostas geradas por pipelines *retrieval‑augmented generation* (RAG).  
Duas delas são as mais frequentemente usadas para medir **a relação entre o contexto (documentos recuperados) e a resposta final** 
produzida pelo modelo:

| Métrica | O que mede | Como é calculada | Por que importa |
|---------|------------|------------------|-----------------|
| **Context Recall** (ou *Context Relevance*) | **Quão bem o conjunto de documentos recuperados cobre as informações necessárias para 
responder à pergunta**. Em outras palavras: “O conteúdo recuperado contém tudo o que eu preciso saber?”. | 1. Seleciona, entre os 
passagens retornadas, aquelas que realmente são usadas pelo modelo na resposta (usando *answer spans* ou avaliando semântica). <br>2. 
Calcula a taxa de cobertura: <br>`Context Recall = (# de informações relevantes encontradas no contexto) / (total de informações 
necessárias da resposta)`. | - Garante que o pipeline não “esqueça” fatos críticos porque eles foram descartados na fase de 
recuperação.<br>- Útil para diagnosticar falhas no *retriever* (o componente que busca documentos). |
| **Faithfulness** (ou *Answer Faithfulness*) | **Quão fielmente a resposta respeita o conteúdo do contexto**. Ela verifica se tudo 
que a resposta afirma está realmente presente nas passagens recuperadas e não introduz informação externa ou inventada. | 1. Extrai 
as frases *claims* da resposta gerada (geralmente via parsing semântico). <br>2. Verifica, para cada claim, se há suporte no texto de 
contexto (usando similaridade vetorial, entailment modelo ou correspondência exata). <br>3. A métrica equivale à proporção de claims 
que recebem *entailed* (“não contradiz nem excede” o contexto). | - Avalia a **qualidade da geração**: evita “hallucinações”, 
sobre‑interpretações ou misturações com fatos não presentes no documento.<br>- É crucial quando se exige respostas seguras e 
auditáveis (ex.: áreas médicas, jurídicas ou regulatórias). |

---

## Explicação mais detalhada

### 1. Context Recall  
- **Objetivo:** Verificar se o *retriever* conseguiu trazer *todas* as passagens relevantes antes que o modelo gerador escreva a 
resposta.  
- **Como funciona na prática (RAGAS):**  
  - Você fornece à métrica um conjunto de “gold documents” — passagens verdadeiramente relevante para a pergunta.  
  - O algoritmo compara essas gold docs com as passagens retornadas pelo seu retriever e calcula:  

    \[
    \text{Context Recall@k} = \frac{\text{Número de gold docs que aparecem nas k primeiras passages recuperadas}}{\text{Tamanho total 
de gold docs}}
    \]

  - Em versões mais avançadas, em vez de comparar documentos inteiros, usa‑se **overlap** semântico (cosine similarity entre 
embeddings) ou identifica **answer spans** dentro das passagens retornadas e verifica se elas cobrem as respostas “de verdade” do 
*ground truth*.  
- **Interpretation típica:** 0.85 → 85 % da informação essencial estava disponível nos documentos candidatos.

### 2. Faithfulness  
- **Objetivo:** Verificar se o modelo de geração não “inventa” nem distorce dados que estão no contexto.  
- **Como funciona na prática (RAGAS):**  
  - Extrai as *claims* (afirmações) da resposta gerada. Exemplo: para a pergunta *“Qual é a capital da Austrália?”*, uma claim pode 
ser “Canberra é a capital”.  
  - Para cada claim, aplica um **verification step** que usa modelos de entailment ou busca por correspondência lexical/semântica no 
texto do contexto recuperado.  
    - *Entailment = True* → claim suportada.  
    - *Entailment = False* (ou “Not Enough Info”) → claim contraditiva ou não justificada.  
  - A métrica final pode ser:  

    \[
    \text{Faithfulness} = \frac{\text{Número de claims suportadas}}{\text{Total de claims geradas}}
    \]

- **Interpretation típica:** 0.92 → 92 % das afirmações da resposta são exatamente sustentadas pelo contexto; o restante indica risco 
de *hallucination*.

---

## Por que comparar as duas?

| Aspecto | Context Recall | Faithfulness |
|--------|----------------|--------------|
| **Foco principal** | Qualidade da **recuperação**. Se a informação importante não está no conjunto retornado, a métrica penaliza o 
retriever. | Qualidade da **geração**. Mesmo que tudo esteja presente, se o modelo inventar algo ou omitir algo obrigatório, Ele é 
penalizado. |
| **Indicador de melhoria** | Melhorar o *retriever* (embeddings, técnicas de re‑ranking). | Melhorar o *generator* (prompt 
engineering, ajuste fino) e/ou ajustar os mecanismos de verificação/validação. |
| **Combinação típica** | Você pode usar as duas métricas juntas para obter um panorama completo: alta recall + alta faithfulness ≈ 
pipeline equilibrado. <br>Se a recall for baixa mas a faithfulness for alta, o problema provavelmente está na fase de busca e não na 
geração. |

---

## Exemplo prático

**Pergunta:** “Qual foi a causa da Revolução Francesa?”

1. **Contexto “gold”** (documentos esperados):  
   - Passagem A: “Em 1789 o povo francês se rebelou contra a monarquia devido a crises financeiras e desigualdades sociais.”  
   - Passagem B: “A tomada da Bastilha em 14 de julho marcou o início do conflicto.”

2. **Recuperado pelo retriever** (k=5):  
   - Apenas a Passagem A foi recuperada; Passagem B não apareceu.

   → *Context Recall* ≈ 0.5 (uma das duas gold passages foi encontrada).

3. **Resposta gerada:** “A Revolução Francesa começou em 1792, quando os revolucionários derrubaram o rei Luís XVIII.”  

   - Esta resposta contém uma factual incorrect (“Luís XVIII” nunca existiu na época).  
   - *Faithfulness* = 0 (a claim “começou em 1792 com a queda do rei Luís XVIII” não está presente nem no contexto recuperado).

Neste caso, o **retriever** falhou ao trazer as passagens completas (baixa recall) e ao mesmo tempo o modelo gerou informação 
incorreta, refletindo baixa faithfulness.

---

## Como usar essas métricas na prática

1. **Defina seu gold standard**  
   - Crie um conjunto de perguntas + pares *question‑answer* verificados por humanos (ou use datasets padrão como HotpotQA, 
TriviaQA).  
2. **Integre à pipeline de avaliação**  
   ```python
   from raagas import evaluate
   evaluation_results = evaluate(
       query_id="q1",
       contexts=[doc_a, doc_b],      # passagens recuperadas
       answer_generated="..."        # resposta do modelo LLM
   )
   print(evaluation_results["context_recall"])
   print(evaluation_results["faithfulness"])
   ```  
3. **Analise os scores**  
   - Se *Context Recall* está acima de 0,8 mas *Faithfulness* é baixa → priorize melhorar o modelo gerador ou adicione mecanismos de 
verificação (e.g., self‑consistency).  
   - Se a *Faithfulness* já está alta, aumente *k* no retriever ou ajuste thresholds de busca para capturar mais gold docs.  
4. **Monitore ao longo do tempo**  
   - Registre os scores em cada versionamento da pipeline; pequenas quedas podem indicar regressões sutis.

---

## Resumo rápido

- **Context Recall**: “O contexto recuperado tem a informação que eu preciso?” → métrica de **completeness/retrieval quality**.  
- **Faithfulness**: “A resposta gerada está *exatamente* contida no contexto?” → métrica de **veracidade/geration quality**.

Ambas são complementares: uma alta em uma não garante qualidade na outra; para um RAG robusto você deve acompanhá‑las simultaneamente 
e fazer ajustes nas duas frentes (retriever + generator).  

Se precisar de exemplos de código, de como montar seu próprio conjunto de avaliação ou de DOs e DON’Ts para melhorar essas métricas, 
é só dizer! 🚀