import os
import sys
from pathlib import Path
import asyncio
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

## bibliotecas Ollama, OpenAI e Ragas
from ollama import Client

from openai import AsyncOpenAI

from ragas.llms import llm_factory

## ContextPrecision está em ragas.metrics.collections e não em ragas.metrics
from ragas.metrics.collections import Faithfulness 
#
# ocultar o servidor
ollama_server = os.environ['OLLAMA_SERVER']
host=f'http://{ollama_server}:11434/v1'
#
## Setup LLM - aqui será o ajuste para Ollama models
client = AsyncOpenAI(
    api_key="ollama",
    base_url=host
)
#
llm = llm_factory("nemotron-3-nano:30b", provider="openai", client=client)

### Create metric
scorer = Faithfulness(llm=llm)
###
print ("___ Scorer___")
print (scorer)
print ("___ Scorer___")
#
###
#### Evaluate
#### Neste caso, o que é comparado são as tres frases "retrieved" contra a "reference" 
#### A ordem importa. Então, como a terceira frase é igual a reference, o score é 0,333.
input = "Why is the sky blue?"
response = "the sky appears blue because of Rayleigh scattering, where shorter-wavelength blue light is scattered in all directions by tiny gas molecules in the atmosphere!"
# retrieved = ["cartão de crédito é um instrumento de crédito pessoal",] 
retrieved = ["1. **Sunlight enters Earth's atmosphere**: When the sun shines, it sends out a vast array of electromagnetic radiation, including all the colors of the visible spectrum (red, orange, yellow, green, blue, indigo, and violet).  \2. **Light scatters off tiny particles**: As sunlight travels through the atmosphere, it encounters tiny molecules of gases like nitrogen (N2) and oxygen (O2). These particles are much smaller than the wavelength of light.  \3. **Blue light is scattered more**: When sunlight hits these tiny particles, some of the shorter-wavelength colors, like blue and violet, are scattered in all directions by the gas molecules. This is known as Rayleigh scattering, named after the British physicist Lord Rayleigh who first described it.  \4. **Red light passes through relatively unscattered**: The longer-wavelength colors, like red and orange, pass through the atmosphere with less scattering because they have a lower frequency and are not affected as much by the tiny particles.",
           "**Blue light is distributed throughout the sky**: Because blue light is scattered more easily, it reaches our eyes from all directions in the sky, giving us the illusion that the sky is blue.",
           "**The other colors are still visible**: Although red and orange light passes through relatively unscattered, they are absorbed or reflected by atmospheric particles, which is why we see a range of colors during sunrise and sunset.",
]

print (" ") 
print ("input: ", input)
print ("response: ", response)
print ("retrieved contexts: ", retrieved)
print (" ")
#
result = scorer.score(
    user_input=input,
    response=response,
    retrieved_contexts=retrieved
     
)
print(f"Faithfulness Score: {result.value}")