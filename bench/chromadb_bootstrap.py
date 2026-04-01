"""
Some chromadb wheels ship an incomplete chromadb/api/types.py while
utils/embedding_functions still import Space, SparseEmbeddingFunction, etc.
Patch the on-disk types file once so `import chromadb` succeeds.

Loaded via importlib from main.py before any other project imports.
"""
from __future__ import annotations

import sys
from pathlib import Path


def _patch_chromadb_api_types() -> None:
    ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    p = Path(sys.prefix) / "lib" / f"python{ver}" / "site-packages" / "chromadb" / "api" / "types.py"
    if not p.is_file():
        return
    text = p.read_text(encoding="utf-8")
    if "PillagerBench chromadb shim end" in text:
        return

    need_space = "Space = Literal" not in text
    need_sparse = "SparseEmbeddingFunction" not in text
    if not need_space and not need_sparse:
        return

    lines = ["", "# --- PillagerBench chromadb shim start ---"]
    if need_space:
        lines.extend(
            [
                "from typing import Literal",
                'Space = Literal["cosine", "l2", "ip"]',
            ]
        )
    if need_sparse:
        lines.extend(
            [
                "from typing import Any, List, Protocol, TypeVar",
                "from typing_extensions import runtime_checkable",
                "_DSp = TypeVar('_DSp')",
                "if 'SparseVectors' not in globals():",
                "    SparseVectors = List[Any]",
                "@runtime_checkable",
                "class SparseEmbeddingFunction(Protocol[_DSp]):",
                "    def __call__(self, input: _DSp) -> Any: ...",
            ]
        )
    lines.append("# --- PillagerBench chromadb shim end ---")
    p.write_text(text.rstrip() + "\n" + "\n".join(lines) + "\n", encoding="utf-8")


_patch_chromadb_api_types()
