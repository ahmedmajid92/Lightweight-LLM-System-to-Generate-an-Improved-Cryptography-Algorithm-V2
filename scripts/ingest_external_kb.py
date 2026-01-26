from __future__ import annotations

from pathlib import Path

from cipherlab.config import load_settings
from cipherlab.rag.documents import load_documents_from_dirs

if __name__ == "__main__":
    settings = load_settings()
    root = Path(settings.project_root)
    ext_dir = root / settings.kb_external_dir
    ext_dir.mkdir(parents=True, exist_ok=True)

    docs = load_documents_from_dirs([ext_dir])
    print(f"Found {len(docs)} external documents in {ext_dir}")
    for d in docs[:25]:
        print(f"- {d.title} ({d.source_path}) chars={len(d.text)}")
    print("Done. Now run: python scripts/build_kb_index.py")
