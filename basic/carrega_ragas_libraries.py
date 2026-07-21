import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from openai import OpenAI
from ragas.llms import llm_factory

host=os.environ['OLLAMA_SERVER']

### Ollama exposes OpenAI-compatible API ###
client = OpenAI(
    api_key="ollama",
    base_url=host
)
llm = llm_factory("llama3.1", provider="openai", client=client)
# Uses Instructor adapter