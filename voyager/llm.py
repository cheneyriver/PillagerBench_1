import logging
import os
import time
from typing import Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
try:
    # Optional dependency: only required if you use `deepseek-*` models.
    from langchain_deepseek import ChatDeepSeek
except ModuleNotFoundError:  # pragma: no cover
    ChatDeepSeek = None
try:
    from langchain_openai import ChatOpenAI
except ModuleNotFoundError:  # pragma: no cover
    ChatOpenAI = None
try:
    from langchain_ollama import ChatOllama
except ModuleNotFoundError:  # pragma: no cover
    ChatOllama = None

logger = logging.getLogger(__name__)


if ChatOpenAI is not None:
    class ChatOpenRouter(ChatOpenAI):
        openai_api_base: str
        openai_api_key: str
        model_name: str

        def __init__(self,
                     model_name: str,
                     api_key: Optional[str] = None,
                     base_url: str = "https://openrouter.ai/api/v1",
                     **kwargs):
            api_key = api_key or os.getenv('OPENROUTER_API_KEY')
            super().__init__(base_url=base_url,
                             api_key=api_key,
                             model_name=model_name, **kwargs)
else:  # pragma: no cover
    ChatOpenRouter = None


def create_llm(
        model_name,
        temperature,
        request_timeout,
) -> BaseChatModel:
    if model_name.startswith("gpt"):
        if ChatOpenAI is None:
            raise ModuleNotFoundError(
                "langchain_openai is not installed. Install it with "
                "`pip install langchain-openai` (or switch to a non-OpenAI model)."
            )
        openai_api_base = os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_API_BASE_URL")
        llm_kwargs = {}
        if openai_api_base:
            # LangChain's ChatOpenAI accepts openai_api_base (alias: base_url)
            llm_kwargs["openai_api_base"] = openai_api_base
        model = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            request_timeout=request_timeout,
            max_retries=10,
            **llm_kwargs,
        )
    elif model_name.startswith("o3") or model_name.startswith("o1"):
        if ChatOpenAI is None:
            raise ModuleNotFoundError(
                "langchain_openai is not installed. Install it with "
                "`pip install langchain-openai` (or switch to a non-OpenAI model)."
            )
        openai_api_base = os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_API_BASE_URL")
        llm_kwargs = {}
        if openai_api_base:
            # LangChain's ChatOpenAI accepts openai_api_base (alias: base_url)
            llm_kwargs["openai_api_base"] = openai_api_base
        model = ChatOpenAI(
            model_name=model_name,
            request_timeout=request_timeout,
            max_retries=10,
            **llm_kwargs,
        )
    elif model_name.startswith("deepseek"):
        if ChatDeepSeek is None:
            raise ModuleNotFoundError(
                "langchain_deepseek is not installed. Install it with "
                "`pip install langchain-deepseek-official` (or switch to a non-deepseek model)."
            )
        model = ChatDeepSeek(
            model_name=model_name,
            temperature=temperature,
            request_timeout=request_timeout,
            max_retries=10,
        )
    elif model_name.startswith("ollama-"):
        if ChatOllama is None:
            raise ModuleNotFoundError(
                "langchain_ollama is not installed. Install it with "
                "`pip install langchain-ollama` (or switch to a non-ollama model)."
            )
        model = ChatOllama(
            model=model_name[7:],
            base_url="host.docker.internal",
            temperature=temperature,
            request_timeout=request_timeout,
            max_retries=10,
            num_ctx=16382,
        )
    elif model_name.startswith("openrouter-"):
        if ChatOpenAI is None:
            raise ModuleNotFoundError(
                "langchain_openai is not installed. Install it with "
                "`pip install langchain-openai` (or switch to a non-openrouter model)."
            )
        model = ChatOpenAI(
            model_name=model_name[11:],
            temperature=temperature,
            request_timeout=request_timeout,
            max_retries=10,
            api_key=os.getenv('OPENROUTER_API_KEY'),
            base_url="https://openrouter.ai/api/v1",
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return model


def invoke_with_log(model: BaseChatModel, messages, *args, prefix="", **kwargs):
    model_name = model.model_name
    if model_name.startswith("o3") or model_name.startswith("o1") or model_name.startswith("openrouter-o3"):
        # Replace system messages with developer messages
        for i, message in enumerate(messages):
            if isinstance(message, SystemMessage):
                messages[i] = {"role": "developer", "content": message.content}
            elif isinstance(message, dict) and message.get("role") == "system":
                messages[i] = {"role": "developer", "content": message.get("content")}

    # Start measuring time
    start_time = time.time()

    # Call the original invoke method
    response = model.invoke(messages, *args, **kwargs)

    # Calculate elapsed time
    elapsed_time = time.time() - start_time

    # Log the time taken and token usage (assuming 'usage' key contains token info)
    logger.info(f"{prefix}AI invoke: response time: {elapsed_time:.2f}s, usage metadata: {response.usage_metadata}")

    return response
