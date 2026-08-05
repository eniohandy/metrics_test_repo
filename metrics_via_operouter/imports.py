### imports required for metrics execution

import os
import sys
import json
from pathlib import Path
import asyncio

# minha função de importacao dos modelos
from model_loader import load_model_names
from data_loader import load_test_cases

from ollama import Client
from openai import AsyncOpenAI
from ragas.llms import llm_factory
