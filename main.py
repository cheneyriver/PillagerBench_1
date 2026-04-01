import importlib.util
import os
from pathlib import Path

_bootstrap = Path(__file__).resolve().parent / "bench" / "chromadb_bootstrap.py"
if _bootstrap.is_file():
    _spec = importlib.util.spec_from_file_location("_pillagerbench_chromadb_bootstrap", _bootstrap)
    if _spec and _spec.loader:
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)

import hydra

from api_keys import *
from bench.config import Config
from bench.pillager_bench import PillagerBench


@hydra.main(config_path="configs", config_name="benchmark", version_base="1.3")
def main(args: Config):
    # set openai api key
    if "openai_api_key" in globals():
        os.environ["OPENAI_API_KEY"] = openai_api_key
    if "openai_api_base" in globals() and openai_api_base:
        os.environ["OPENAI_API_BASE"] = openai_api_base
    if not os.environ.get("OPENAI_API_BASE"):
        os.environ["OPENAI_API_BASE"] = "https://api.chatanywhere.tech/v1"
    # If you use an OpenAI-compatible endpoint (e.g., ChatAnywhere),
    # set openai_api_base in api_keys.py or OPENAI_API_BASE (voyager/llm.py -> ChatOpenAI).
    if "deepseek_api_key" in globals():
        os.environ["DEEPSEEK_API_KEY"] = deepseek_api_key
    if "openrouter_api_key" in globals():
        os.environ["OPENROUTER_API_KEY"] = openrouter_api_key

    bench = PillagerBench(args)
    bench.run()

if __name__ == "__main__":
    main()
