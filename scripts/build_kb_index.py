from __future__ import annotations

from cipherlab.config import load_settings
from cipherlab.rag.index import build_kb_index

if __name__ == "__main__":
    settings = load_settings()
    build_kb_index(settings)
    print("KB index built successfully.")
