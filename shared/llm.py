"""
Shared LLM configuration.

Default for LangGraph notebooks (07–09): DeepSeek V4 Flash via OpenAI-compatible API.
Switch providers with LLM_PROVIDER in `.env`.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

_THIS_DIR = Path(__file__).resolve().parent
_ENV_FILE = _THIS_DIR.parent / ".env"
load_dotenv(dotenv_path=_ENV_FILE)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")

ZAI_API_KEY = os.getenv("ZAI_API_KEY", "")
ZAI_BASE_URL = os.getenv("ZAI_BASE_URL", "https://api.z.ai/api/paas/v4")

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "deepseek").lower()


def get_model():
    """
    Returns a configured chat model instance.

    Supported LLM_PROVIDER values:
        - deepseek (default): DeepSeek V4 Flash
        - zai: z.ai GLM-5 via OpenAI-compatible endpoint
        - openai: OpenAI models via init_chat_model
        - anthropic: Anthropic Claude via init_chat_model
        - openrouter: Any model on OpenRouter via init_chat_model

    Override with environment variables in `.env` — never commit real keys.
    """
    if LLM_PROVIDER == "deepseek":
        from langchain_openai import ChatOpenAI

        if not DEEPSEEK_API_KEY:
            raise ValueError(
                "DEEPSEEK_API_KEY is not set. Copy .env.example to .env and add your DeepSeek key, "
                "or set LLM_PROVIDER to another provider with its matching API key."
            )

        return ChatOpenAI(
            model=DEEPSEEK_MODEL,
            api_key=DEEPSEEK_API_KEY,  # type: ignore[arg-type]
            base_url=DEEPSEEK_BASE_URL,
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.3")),
            max_retries=int(os.getenv("LLM_MAX_RETRIES", "6")),
            # Disable thinking so tool-calling + structured output work on DeepSeek V4.
            # JSON mode alternative: https://api-docs.deepseek.com/guides/json_mode
            extra_body={"thinking": {"type": "disabled"}},
        )

    if LLM_PROVIDER == "openai":
        from langchain.chat_models import init_chat_model

        return init_chat_model(os.getenv("OPENAI_MODEL", "openai:gpt-4.1"))

    if LLM_PROVIDER == "anthropic":
        from langchain.chat_models import init_chat_model

        return init_chat_model(os.getenv("ANTHROPIC_MODEL", "anthropic:claude-sonnet-4-6"))

    if LLM_PROVIDER == "openrouter":
        from langchain.chat_models import init_chat_model

        model = os.getenv("OPENROUTER_MODEL", "openrouter:anthropic/claude-sonnet-4-6")
        return init_chat_model(model)

    # z.ai GLM-5
    from langchain_openai import ChatOpenAI

    if not ZAI_API_KEY:
        raise ValueError(
            "ZAI_API_KEY is not set. Copy .env.example to .env and add your key, "
            "or set LLM_PROVIDER=deepseek with DEEPSEEK_API_KEY."
        )

    return ChatOpenAI(
        model=os.getenv("ZAI_MODEL", "glm-5"),
        api_key=ZAI_API_KEY,  # type: ignore[arg-type]
        base_url=ZAI_BASE_URL,
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
        max_retries=int(os.getenv("LLM_MAX_RETRIES", "6")),
    )
