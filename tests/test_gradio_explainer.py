from pathlib import Path

import gradio_explainer_ui.app as explainer


def test_build_demo_smoke():
    demo = explainer.build_demo()
    assert demo.__class__.__name__ == "Blocks"


def test_framework_flowchart_paths_are_logically_separated():
    svg = explainer._framework_flow_svg()
    assert 'x1="360" y1="377" x2="320" y2="377"' not in svg
    assert 'x1="548" y1="377" x2="500" y2="377"' not in svg
    assert 'M 612 323 C 560 286, 418 286, 280 328' in svg
    assert 'x1="548" y1="411" x2="500" y2="411"' in svg
    assert 'x1="374" y1="596" x2="430" y2="596"' in svg
    assert 'M 374 584 C 500 548, 636 500, 780 410' in svg


def test_gradio_keeps_thesis_facing_model_labels():
    source = Path(explainer.__file__).read_text(encoding="utf-8")
    assert "DeepSeek-R1-Distill-Qwen-7B" in source
    assert "Qwen2.5-Coder-7B-Instruct" in source
    assert "DeepSeek-R1-Distill-Qwen-14B" not in source
    assert "DeepSeek-Coder-V2-Lite-Instruct" not in source
    assert "deepseek/deepseek-r1" not in source
    assert "deepseek/deepseek-chat-v3-0324" not in source
