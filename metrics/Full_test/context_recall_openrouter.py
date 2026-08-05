"""
Context Recall via OpenRouter
==============================
Adaptação do script original Ollama para usar modelos pagos via OpenRouter.
Uso:
    export OPENROUTER_API_KEY=sk-or-...
    python context_recall_openrouter.py --models-file models.txt --data-file data.json
"""

import os
import sys
import json
from pathlib import Path
import asyncio
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextRecall

import argparse
from model_loader import load_model_names
from data_loader import load_test_cases

from dotenv import load_dotenv
load_dotenv()

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--models-file", type=str, required=True)
parser.add_argument("--data-file",   type=str, required=True)
args = parser.parse_args()

model_names = load_model_names(args.models_file)
test_cases  = load_test_cases(args.data_file)

# ── OpenRouter client ─────────────────────────────────────────────────────────
# diferença 1: chave vem de OPENROUTER_API_KEY
# diferença 2: base_url aponta para OpenRouter
openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
if not openrouter_api_key:
    raise EnvironmentError("Variável OPENROUTER_API_KEY não definida.")

client = AsyncOpenAI(
    api_key=openrouter_api_key,
    base_url="https://openrouter.ai/api/v1"
)

# ── Avaliação ─────────────────────────────────────────────────────────────────
for model_name in model_names:
    print("___ Modelo ___")
    print(f"=== {model_name} ===")

    # provider="openai" se mantém — OpenRouter é compatível com o protocolo OpenAI
    llm    = llm_factory(model_name, provider="openai", client=client)
    scorer = ContextRecall(llm=llm)

    print("___ Scorer ___")
    print(scorer)
    print("______")

    for case in test_cases:
        result = scorer.score(
            user_input=case["user_input"],
            reference=case["reference"],
            retrieved_contexts=case["retrieved_contexts"]
        )

        print("___ Pontuação ___")
        print(f"Context Recall Score: {result.value}")
        print("______")
