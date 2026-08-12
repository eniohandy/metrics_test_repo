# Investigação de Métricas RAGAS em Ambiente Local (Ollama)

## Relatório de testes, cenários e discussões

---

## Sumário

1. Introdução e objetivo
2. Capítulo 1 — As métricas
3. Capítulo 2 — Infraestrutura e ambiente de testes
4. Capítulo 3 — Cenários de teste
5. Capítulo 4 — Resultados dos testes
6. Capítulo 5 — Discussão: subjetividade e limites metodológicos
7. Conclusões

---

## 1. Introdução e objetivo

Esta investigação partiu de uma dúvida prática: como funcionam, na prática, as métricas de avaliação de sistemas RAG (Retrieval-Augmented Generation) da biblioteca **ragas**, e o quanto seus resultados são confiáveis e reprodutíveis quando calculadas por diferentes modelos de LLM rodando localmente via **Ollama**.

O trabalho evoluiu em etapas:
- Entendimento manual das fórmulas de `FactualCorrectness`, `ContextRecall`, `ContextPrecision` e `Faithfulness`.
- Resolução de uma série de problemas de infraestrutura (clients síncronos vs. assíncronos, embeddings, versões do Ollama, drivers de GPU).
- Criação de cenários de teste no domínio financeiro (cartão de crédito, empréstimos, financiamento de veículo), desenhados propositalmente para forçar diferentes tipos de falha.
- Execução desses cenários em 5 modelos locais (`granite4.1:8b`, `cogito:latest`, `gpt-oss:latest`, `nemotron-3-nano:latest`, `llama3.1:latest`) e, ao final, também no Claude (via API), para comparação.
- Discussão crítica sobre até que ponto essas métricas medem algo objetivo, ou se são fundamentalmente um julgamento subjetivo do LLM avaliador.

---

## 2. Capítulo 1 — As métricas

### 2.1 Mecânica comum

`FactualCorrectness`, `ContextRecall` e `Faithfulness` compartilham o mesmo mecanismo de fundo:

```
decompor_em_claims(texto) → verificar_cada_claim_contra_um_texto_de_referência (NLI) → agregar em TP / FP / FN
```

A diferença entre elas está em **quais dois campos são comparados**:

| Métrica | Texto decomposto em claims | Comparado contra | O que avalia |
|---|---|---|---|
| `Faithfulness` | `response` | `retrieved_contexts` | Se o LLM gerador alucinou (inventou algo fora do contexto) |
| `ContextRecall` (LLM) | `reference` | `retrieved_contexts` | Se o retriever trouxe contexto suficiente para sustentar a resposta ideal |
| `FactualCorrectness` | `response` **e** `reference` (nos dois sentidos) | um contra o outro | Sobreposição entre o que foi dito e o que deveria ter sido dito |

Um achado importante do processo: se `response` e `reference` forem o mesmo texto (ou muito parecidos), `Faithfulness` e `ContextRecall` produzem **o mesmo valor numérico**, porque são literalmente a mesma conta rodando sobre a mesma entrada. A diferença conceitual entre as duas métricas só se manifesta quando `response` diverge de `reference`.

### 2.2 TP, FP e FN — o que significam aqui

Diferente de classificação binária clássica (onde um único exemplo não pode ser TP e FN ao mesmo tempo), aqui trabalha-se com **comparação de conjuntos** — parecido com avaliação de extração de entidades (NER):

- **TP**: uma claim do texto avaliado que é confirmada pelo texto de referência.
- **FP**: uma claim do texto avaliado que não é confirmada (ou é contradita).
- **FN**: uma claim do texto de referência que não aparece (ou não é confirmada) no texto avaliado.

Não existe **TN** nessas métricas — não há um universo fechado de "todos os fatos possíveis" contra o qual contar quantos foram corretamente rejeitados.

Um mesmo "erro de fato" pode gerar simultaneamente 1 FP (no conjunto do texto avaliado) e 1 FN (no conjunto do texto de referência) — são elementos de listas diferentes, não o mesmo dado contado duas vezes.

### 2.3 Context Precision — a exceção com ranking

`ContextPrecision` não decompõe nenhum texto em claims. Em vez disso, avalia **cada item de `retrieved_contexts` individualmente**, perguntando "esse contexto é relevante para sustentar o `reference`?" — e agrega o resultado usando uma fórmula ponderada por **posição** (Average Precision, emprestada de sistemas de busca/IR):

$$Precision = \frac{\sum_{k} (Precision@k \times v_k)}{\text{total de itens relevantes}}$$

onde `v_k` é 1 se o item na posição k é relevante, e `Precision@k` é a proporção de itens relevantes entre os primeiros k.

Essa fórmula penaliza pouco um contexto irrelevante que aparece **no fim** da lista, e penaliza muito um contexto irrelevante que aparece **no início** — uma suposição herdada de motores de busca (onde o usuário lê resultados em ordem e para antes do fim), que **pode não se aplicar** a pipelines RAG onde o LLM gerador recebe todos os contextos de uma vez, sem processá-los sequencialmente.

### 2.4 AnswerCorrectness e a dependência de embeddings

Diferente das métricas acima, `AnswerCorrectness` (e `AnswerSimilarity`) também usam um **modelo de embeddings** para calcular similaridade semântica entre `response` e `reference`, além do julgamento por LLM. Isso introduz uma fonte adicional de variabilidade — a qualidade do embeddings model (especialmente sua cobertura de idiomas diferentes do inglês) pode influenciar o resultado, independentemente da qualidade do LLM avaliador.

### 2.5 Por que `.reason` costuma vir `None`

O campo `.reason` do objeto `MetricResult` retornado por `scorer.ascore(...)` só é populado quando a métrica é resultado de **uma única chamada de LLM** com um campo de justificativa embutido (como em `AspectCritic` ou métricas customizadas via `@discrete_metric`). Métricas multi-etapa como `FactualCorrectness`, `ContextRecall` e `Faithfulness` não expõem uma explicação única agregada — a "explicação" real estaria espalhada pelas decisões intermediárias (decomposição + veredictos por claim), que não são expostas diretamente no `MetricResult` padrão.

---

## 3. Capítulo 2 — Infraestrutura e ambiente de testes

A execução local via Ollama exigiu resolver uma série de problemas de infraestrutura, documentados aqui por serem replicáveis em outros ambientes:

### 3.1 Client síncrono vs. assíncrono

O `ragas.evaluate()`, `scorer.ascore()` e até o aparentemente síncrono `scorer.score()` rodam de forma **assíncrona internamente** (via `asyncio`). Isso exige que o client passado ao `llm_factory` e ao `embedding_factory` seja `AsyncOpenAI`, não `OpenAI` — usar o client síncrono produz o erro `TypeError: Cannot use aembed_text() with a synchronous client`.

### 3.2 Embeddings obrigatórios para `AnswerCorrectness`

Métricas que calculam similaridade semântica (`AnswerCorrectness`) exigem um `embeddings` explícito. Sem isso, o `ragas` tenta criar um default que assume credenciais OpenAI reais, causando `openai.OpenAIError: Missing credentials`. A correção é criar o embeddings explicitamente via `embedding_factory(provider="openai", model="nomic-embed-text", client=client)` e passá-lo à métrica (`AnswerCorrectness(llm=llm, embeddings=embeddings)`).

### 3.3 Versão do Ollama e regressão de performance

Durante os testes, identificou-se que a versão mais recente do Ollama (linha 0.32.x) rodava **visivelmente mais devagar** que uma versão anterior (0.15.2) no mesmo hardware (2x RTX 3090). Essa observação é consistente com relatos documentados publicamente de regressões de performance entre versões do Ollama, frequentemente ligadas a incompatibilidade entre o driver NVIDIA instalado (535.309.01) e o runtime CUDA esperado pelas versões mais novas do engine.

### 3.4 Erro de `max_tokens`

Em cenários com muitos `retrieved_contexts` (como o cenário de financiamento com 10 contextos), alguns modelos (`nemotron-3-nano`, `gpt-oss`) produziram o erro `IncompleteOutputException: The output is incomplete due to a max_tokens length limit` — a resposta estruturada (JSON de claims + veredictos) foi cortada antes de terminar. A correção foi elevar o parâmetro `max_tokens` no `llm_factory` (de um default insuficiente, ~1024, para 4096 ou mais).

### 3.5 Automação de testes em lote

Para testar múltiplos modelos e múltiplos cenários de forma sistemática, o fluxo foi automatizado com:
- Um arquivo `models.json` listando os modelos a testar (gerado via `curl .../api/tags | jq`).
- Um (ou mais) arquivo(s) de cenários no formato `{"user_input": ..., "reference": ..., "retrieved_contexts": [...]}`.
- Um script Python que itera sobre modelos × cenários, instanciando um novo `llm` por modelo e rodando cada métrica sobre cada cenário.

---

## 4. Capítulo 3 — Cenários de teste

Foram construídos cenários no domínio financeiro (cartão de crédito, empréstimo pessoal, atendimento bancário, financiamento de veículo), cada um desenhado para forçar um tipo específico de falha, mantendo o template mínimo `{user_input, reference, retrieved_contexts}`.

### 4.1 Conjunto pequeno (4 cenários)

| Cenário | Descrição | Problema forçado |
|---|---|---|
| **1 — Controle** (cartão de crédito) | Reference com 2 claims, todas bem cobertas por contextos relevantes, 1 contexto irrelevante no final | Nenhum — deve pontuar alto em tudo |
| **2 — Recall incompleto** (empréstimo pessoal) | Reference com 3 claims (prazo, taxa, aprovação); contextos cobrem só prazo e aprovação, faltando a taxa | Context Recall baixo |
| **3 — Contradição** (horário bancário) | Reference afirma horário (10h–16h) que diverge do contexto recuperado (9h–18h) | Recall/consistência baixa |
| **4 — Ruído** (aumento de limite de cartão) | Contextos relevantes misturados com 2 trechos totalmente fora do assunto (fundação do banco, previsão do tempo) | Context Precision baixo |

### 4.2 Cenário grande (financiamento de veículo)

Um cenário com **5 claims** no reference (prazo, taxa, entrada mínima, seguro, aprovação online) e **10 retrieved_contexts**, intercalando:
- 4 contextos que sustentam diretamente 4 das 5 claims (posições 1, 3, 5, 7)
- 1 contexto que **contradiz** a claim do seguro obrigatório (afirma que é opcional)
- 4 contextos de ruído puro (fundação do banco, previsão do tempo, cashback do cartão, rendimento da poupança)

Esse cenário foi desenhado especificamente para testar a sensibilidade do `ContextPrecision` ao **ranking** — os contextos relevantes vêm intercalados com ruído, não agrupados no início.

### 4.3 Versão em inglês

Os 4 cenários pequenos e o cenário grande foram traduzidos integralmente para o inglês, mantendo a mesma estrutura semântica (mesmas claims, mesmas contradições, mesmo ruído nas mesmas posições), com o objetivo de investigar se o idioma do texto afeta a qualidade do julgamento dos diferentes modelos.

---

## 5. Capítulo 4 — Resultados dos testes

### 5.1 Gabarito calculado manualmente (4 cenários pequenos)

| Cenário | Context Recall | Context Precision (ranking) | Faithfulness (proxy via reference) |
|---|---|---|---|
| 1 — Controle | 1,00 | 1,00 | 1,00 |
| 2 — Recall incompleto | 0,667 | 1,00 | 0,667 |
| 3 — Contradição | 0,00* | 0,00* | 0,00* |
| 4 — Ruído | 1,00 | 1,00 | 1,00 |

*\*Ver revisão desse gabarito na Seção 6.2 — o valor 0,00 foi posteriormente questionado.*

### 5.2 Gabarito do cenário grande (financiamento de veículo)

| Métrica | Valor esperado |
|---|---|
| Context Recall | 0,80 (4 de 5 claims suportadas) |
| Context Precision (ponderada por ranking) | ≈0,71 |
| Context Precision (razão simples, para comparação) | ≈0,44 |
| Faithfulness (proxy) | 0,80 |

### 5.3 Context Precision — resultados por modelo (português)

| Cenário | granite4.1:8b | cogito:latest | gpt-oss:latest | nemotron-3-nano:latest | llama3.1:latest |
|---|---|---|---|---|---|
| 1 | 0,50 → 0,58 | 1,00 | 1,00 | 0,50 | 0,83 → 1,00 |
| 2 | 0,00 | 0,00 → 1,00 | 1,00 | 1,00 | 0,00 → 1,00 |
| 3 | 0,00 | 0,33 → 0,00 | 0,00 | 0,00 | 0,00 |
| 4 | 1,00 | 1,00 | 1,00 | 1,00 | 1,00 |

*(valores com seta indicam divergência entre duas rodadas do mesmo cenário/idioma/modelo, sem parâmetros de temperatura fixados)*

### 5.4 Context Recall — resultados por modelo (português)

| Cenário | granite4.1:8b | cogito:latest | gpt-oss:latest | nemotron-3-nano:latest | llama3.1:latest |
|---|---|---|---|---|---|
| 1 | 1,00 | 0,50 → 1,00 | 1,00 | 1,00 | 1,00 |
| 2 | 0,667 → 1,00 | 0,50 → 0,667 | 0,667 | 0,667 | 0,75 → 1,00 |
| 3 | 0,00 | 0,00 | 0,00 | 1,00 → 0,00 | 0,00 → 0,50 |
| 4 | 1,00 | 0,50 | 1,00 | 1,00 | 0,50 → 0,40 |

### 5.5 Faithfulness — resultados por modelo (português, rodada completa)

| Cenário | granite4.1:8b | cogito:latest | gpt-oss:latest | nemotron-3-nano:latest | llama3.1:latest |
|---|---|---|---|---|---|
| 1 | 1,00 | 0,667 | 1,00 | 1,00 | 1,00 |
| 2 | 0,667 | 0,667 | 0,667 | 0,667 | 0,667 |
| 3 | 0,50 | 0,50 | 0,00 | 0,50 | 0,50 |
| 4 | 1,00 | 1,00 | 1,00 | 1,00 | 0,50 |

### 5.6 Claude Sonnet 5 (via API) — Faithfulness

| Cenário | Claude Sonnet 5 |
|---|---|
| 1 | 1,00 |
| 2 | 0,667 |
| 3 | 0,50 |
| 4 | 1,00 |

Resultado alinhado ao consenso majoritário dos modelos locais (4 de 5 modelos convergindo em 0,50 no Cenário 3), e não ao gabarito manual original (0,00) — ver discussão na Seção 6.2.

### 5.7 Erros técnicos observados

- **`IncompleteOutputException` (max_tokens)**: ocorreu de forma recorrente em `nemotron-3-nano` e pontualmente em `gpt-oss`, sobretudo em cenários com mais contextos/claims. Corrigido elevando `max_tokens` no `llm_factory`.
- **`nan` em `AnswerCorrectness`**: causado por embeddings não configurado explicitamente, ou por client síncrono sendo usado num fluxo assíncrono — ambos descritos na Seção 3.

---

## 6. Capítulo 5 — Discussão: subjetividade e limites metodológicos

### 6.1 Variância entre rodadas idênticas

Um achado central do experimento foi que **o mesmo modelo, no mesmo cenário, no mesmo idioma, sem nenhuma mudança de entrada, produziu valores diferentes entre rodadas** — por exemplo, `cogito` no Cenário 2 (Context Precision) foi de 0,00 para 1,00; `nemotron-3-nano` no Cenário 3 (Context Recall) foi de 1,00 para 0,00.

Isso levanta a questão de fixar `temperature`/`seed` para reduzir essa variância. A posição adotada na investigação foi de que **a decisão depende do objetivo do experimento**:
- Para medir a **confiabilidade geral de um modelo como juiz** em condições realistas de uso, não faz sentido fixar parâmetros — a própria variância entre execuções é o dado relevante a ser medido.
- Para **isolar o efeito de uma variável específica** (como o idioma do texto, PT vs. EN), fixar `temperature=0` é justificável, pois reduz a chance de uma diferença observada ser atribuída erroneamente ao idioma quando na verdade é apenas ruído de amostragem entre execuções.

O `ragas`/`llm_factory` aceita `temperature` como parâmetro (`llm_factory(model, provider="openai", client=client, temperature=0.0)`), mas não há confirmação de que `seed` seja propagado da mesma forma através da cadeia `ragas → instructor → client`.

### 6.2 O caso do Cenário 3: quando o próprio gabarito estava errado

O gabarito manual do Cenário 3 (contradição de horário bancário) previa Faithfulness/Recall = 0,00, tratando a frase *"as agências funcionam de segunda a sexta, das 10h às 16h"* como uma única claim indivisível, contradita integralmente pelo contexto.

Na prática, a maioria dos modelos testados (incluindo o Claude, testado posteriormente via API) convergiu em **0,50** — sugerindo que decompuseram a frase em duas claims separadas: "funciona de segunda a sexta" (suportada pelo contexto) e "o horário é 10h–16h" (não suportada). Sob essa decomposição, a conta 1 TP + 1 FN = 0,50 é matematicamente correta.

Esse episódio é relevante por dois motivos:
1. **Mesmo o autor do cenário**, aplicando o mesmo critério conceitual, chegou a um número diferente do que o processo formal de decomposição produziu — evidenciando que a granularidade da decomposição em claims não é um dado objetivo do texto, é uma escolha interpretativa.
2. Ao mesmo tempo, um **julgamento humano intuitivo/holístico** da frase tende a concordar com o gabarito original (0,00): uma pessoa lendo a frase interpreta "dias + horário" como uma única informação útil e indivisível — se o horário está errado, a informação inteira é considerada errada, não "50% certa".

### 6.3 Duas noções de "correção" em conflito

A discussão expõe uma tensão de fundo entre:

- **Correção atômica** (o que as métricas do ragas de fato calculam): fração de proposições verificáveis individualmente que são sustentadas pela referência/contexto.
- **Correção holística** (o que um avaliador humano tende a julgar): se a informação, como um todo funcional, é útil e confiável para quem a lê.

Essas duas noções coincidem em muitos casos, mas divergem justamente nos casos mais interessantes — onde componentes de uma frase estão semanticamente interligados (como dias + horário de funcionamento), e uma decomposição atômica “fatia” uma informação que, na prática de uso, não faz sentido fatiar.

### 6.4 A suposição de ranking do Context Precision

A fórmula de `ContextPrecision` (Average Precision, ponderada por posição) foi originalmente desenhada para avaliação de motores de busca, sob a suposição de que o usuário lê resultados em ordem e tende a não chegar até o fim da lista — por isso, um item irrelevante no topo penaliza mais que um item irrelevante no final.

Essa suposição **não necessariamente se sustenta** em pipelines de RAG onde o LLM gerador recebe todos os `retrieved_contexts` de uma vez, sem processá-los sequencialmente nem parar antes do fim. Nesses casos, a posição de um contexto irrelevante é irrelevante para o impacto real no sistema — o que importaria seria apenas a proporção de contextos relevantes vs. irrelevantes (uma métrica de precisão simples, sem ponderação por posição).

A ponderação por ranking só se justifica plenamente quando há uma etapa de **truncamento** no pipeline (por exemplo, usar apenas os top-k contextos antes de enviar ao LLM gerador) — nesse caso, itens fora do top-k de fato nunca chegam ao gerador, e a posição volta a ser relevante.

### 6.5 Síntese da crítica metodológica

As métricas do `ragas` não são medições no sentido de instrumentos objetivos e reprodutíveis (como uma régua ou termômetro). São mais próximas de **um painel de avaliadores (LLMs) julgando um texto**, cada um aplicando critérios de decomposição e julgamento ligeiramente diferentes — e, como demonstrado, sujeitos a variância mesmo entre execuções idênticas do mesmo avaliador.

Isso não invalida o uso dessas métricas, mas muda a forma correta de interpretá-las:
- Um único valor de uma única execução deve ser tratado com cautela.
- Comparações entre modelos, idiomas ou versões de sistema devem, idealmente, ser feitas sobre **médias de múltiplas execuções**, não execuções isoladas.
- Divergências sistemáticas entre o julgamento da métrica e o julgamento humano holístico (como no Cenário 3) devem ser documentadas como uma limitação de validade da métrica, não apenas como "erro" de um modelo específico.

---

## 7. Conclusões

1. As métricas `ContextPrecision`, `ContextRecall` e `Faithfulness` compartilham um mecanismo comum de decomposição em claims + verificação NLI, diferindo apenas em quais dois campos são comparados entre si.

2. A infraestrutura local via Ollama exigiu resolver múltiplos problemas técnicos (client assíncrono, embeddings explícitos, versão do engine, `max_tokens`) antes que resultados confiáveis pudessem ser obtidos — problemas que, se não identificados, poderiam ser confundidos com falhas de julgamento dos modelos.

3. Os 5 modelos locais testados (`granite4.1:8b`, `cogito`, `gpt-oss`, `nemotron-3-nano`, `llama3.1`) apresentaram variância significativa entre si e, mais notavelmente, **entre execuções repetidas do mesmo modelo sob as mesmas condições**, sem nenhum parâmetro de determinismo fixado.

4. Testado posteriormente via API, o Claude convergiu com o consenso majoritário dos modelos locais no cenário de maior divergência (Cenário 3), e não com o gabarito manual original — episódio que expôs uma falha do próprio gabarito, não dos modelos avaliados.

5. A investigação aponta para uma limitação metodológica de fundo: essas métricas operacionalizam noções de "correção" e "relevância" através de escolhas de design (granularidade da decomposição em claims, ponderação por posição no ranking) que carregam suposições específicas — nem sempre alinhadas com o julgamento humano holístico, nem sempre alinhadas com o comportamento real de um pipeline de RAG que não processa contextos sequencialmente.

6. Isso não invalida o uso do `ragas` como ferramenta de avaliação, mas recomenda cautela na interpretação de valores isolados, e sugere que resultados sejam sempre acompanhados de múltiplas execuções, múltiplos modelos avaliadores, e, sempre que possível, validação cruzada com julgamento humano em casos de fronteira.

---

## Anexo — Arquivos de cenários gerados durante a investigação

- `test_cases_financeiro.json` — 4 cenários pequenos em português
- `test_cases_financial_en.json` — os mesmos 4 cenários traduzidos para inglês
- `test_case_financiamento_grande.json` — cenário grande (5 claims, 10 contextos) em português
- `test_case_vehicle_financing_large_en.json` — o cenário grande traduzido para inglês
