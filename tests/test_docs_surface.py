from pathlib import Path


DOC_FILES = [
    Path("README.md"),
    Path("kb/10_reproducibility.md"),
    Path("kb/11_dataset_design_for_tuning.md"),
    Path("dataset_card.md"),
    Path("data/sft/dataset_card.md"),
]

FORBIDDEN_TERMS = [
    "OpenAI",
    "OpenRouter",
    "OPENROUTER_API_KEY",
    "deepseek/deepseek-r1",
    "deepseek/deepseek-chat-v3-0324",
    "Gemini",
    "Vertex",
]


def test_docs_keep_thesis_facing_labels_without_runtime_ids():
    combined = "\n".join(path.read_text(encoding="utf-8") for path in DOC_FILES)
    assert "DeepSeek-R1-Distill-Qwen-14B" in combined
    assert "DeepSeek-Coder-V2-Lite-Instruct" in combined
    for term in FORBIDDEN_TERMS:
        assert term not in combined
