"""
LLM provider abstraction - supports Groq (free tier) and HuggingFace Inference API.
"""

import logging
import os
from typing import List, Dict, Optional

from config import CONFIG

logger = logging.getLogger(__name__)

GROQ_MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile",
    "deepseek-r1-distill-llama-70b",
    "gemma2-9b-it",
    "mixtral-8x7b-32768",
]

TOGETHER_MODELS = [
    "mistralai/Mistral-7B-Instruct-v0.2",
    "meta-llama/Llama-3-8b-chat-hf",
]

OLLAMA_MODELS = [
    "llama3",
    "mistral",
    "gemma",
    "phi3",
]


class LLMProvider:
    """
    Unified LLM interface supporting:
    - Groq (free, fast, llama3 / mixtral)
    - HuggingFace Inference API (free tier)
    - Together AI (free credits)
    - Ollama (local, offline)
    """

    def __init__(self):
        self._llm = None
        self._provider = None

    def _load_groq(self, model_name: str, api_key: str):
        from langchain_groq import ChatGroq
        self._llm = ChatGroq(
            groq_api_key=api_key,
            model_name=model_name,
            temperature=CONFIG.llm.temperature,
            max_tokens=CONFIG.llm.max_tokens,
            streaming=True,  # Enables token-by-token streaming
        )
        self._provider = "groq"
        logger.info(f"Groq LLM loaded: {model_name}")

    def _load_huggingface(self, model_name: str, api_key: str):
        from langchain_community.llms import HuggingFaceEndpoint
        self._llm = HuggingFaceEndpoint(
            repo_id=model_name,
            huggingfacehub_api_token=api_key,
            temperature=CONFIG.llm.temperature,
            max_new_tokens=CONFIG.llm.max_tokens,
        )
        self._provider = "huggingface"
        logger.info(f"HuggingFace LLM loaded: {model_name}")

    def _load_ollama(self, model_name: str):
        from langchain_community.chat_models import ChatOllama
        self._llm = ChatOllama(
            model=model_name,
            temperature=CONFIG.llm.temperature,
        )
        self._provider = "ollama"
        logger.info(f"Ollama LLM loaded: {model_name}")

    def initialize(
        self,
        provider: str = None,
        model_name: str = None,
        api_key: str = None,
    ) -> bool:
        """Initialize the LLM. Returns True on success."""
        provider = provider or CONFIG.llm.provider
        model_name = model_name or CONFIG.llm.model_name

        try:
            if provider == "groq":
                key = api_key or CONFIG.llm.groq_api_key or os.getenv("GROQ_API_KEY")
                if not key:
                    raise ValueError("Groq API key not provided.")
                self._load_groq(model_name, key)
            elif provider == "huggingface":
                key = api_key or CONFIG.llm.huggingface_api_key or os.getenv("HUGGINGFACE_API_KEY")
                if not key:
                    raise ValueError("HuggingFace API key not provided.")
                self._load_huggingface(model_name, key)
            elif provider == "ollama":
                self._load_ollama(model_name)
            else:
                raise ValueError(f"Unknown provider: {provider}")

            return True

        except Exception as e:
            logger.error(f"LLM initialization failed: {e}")
            self._llm = None
            raise

    def generate(
        self,
        prompt: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Generate a response given a prompt and optional chat history."""
        if self._llm is None:
            raise RuntimeError("LLM not initialized. Call initialize() first.")

        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

        messages = []
        if chat_history:
            for turn in chat_history:
                if turn["role"] == "user":
                    messages.append(HumanMessage(content=turn["content"]))
                elif turn["role"] == "assistant":
                    messages.append(AIMessage(content=turn["content"]))

        messages.append(HumanMessage(content=prompt))

        response = self._llm.invoke(messages)
        return response.content if hasattr(response, "content") else str(response)

    @property
    def is_ready(self) -> bool:
        return self._llm is not None

    @property
    def provider_name(self) -> str:
        return self._provider or "None"