# llm_utils.py
# --------------
# Wraps the Llama 3.3 API (via OpenAI) into a simple request_llm function.

import os
import openai

# Make sure you have set: export OPENAI_API_KEY="…"
openai.api_key = os.getenv("OPENAI_API_KEY")

def request_llm(model: str, prompt: str, temperature: float = 0.0) -> str:
    """
    Sends `prompt` to the LLM identified by `model` and returns its text reply.
    By default uses temperature=0 for deterministic outputs.
    """
    resp = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )
    return resp.choices[0].message.content.strip()
