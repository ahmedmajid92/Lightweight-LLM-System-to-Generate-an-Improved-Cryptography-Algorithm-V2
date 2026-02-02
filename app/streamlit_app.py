from __future__ import annotations

import json
from typing import Dict, List, Optional

import streamlit as st

from cipherlab.config import load_settings
from cipherlab.cipher.metrics import evaluate_and_score, heuristic_issues
from cipherlab.cipher.registry import ComponentRegistry
from cipherlab.cipher.spec import CipherSpec
from cipherlab.cipher.validator import validate_spec
from cipherlab.cipher.exporter import export_cipher_module
from cipherlab.rag.retriever import RAGRetriever
from cipherlab.utils.repro import make_run_dir, write_json, write_text
from cipherlab.llm.assistant import suggest_improvements
from cipherlab.llm.openai_provider import OpenAIProvider


st.set_page_config(page_title="Crypto Cipher Lab v2 (OpenAI)", layout="wide")

settings = load_settings()

st.title("Crypto Cipher Lab v2 — Block Cipher Builder (OpenAI)")
st.caption("Research-only lab: compose SPN/Feistel/ARX ciphers, run local metrics, and iterate improvements with hybrid RAG.")

registry = ComponentRegistry()

# ---------- Sidebar: OpenAI + RAG settings ----------
st.sidebar.header("OpenAI settings")
api_key = st.sidebar.text_input("OPENAI_API_KEY", value=settings.openai_api_key or "", type="password")
model_fast = st.sidebar.text_input("Fast model", value=settings.openai_model_fast)
model_quality = st.sidebar.text_input("Quality model", value=settings.openai_model_quality)
model_for_improve = st.sidebar.selectbox("Model for improvement suggestions", [model_fast, model_quality], index=0)

st.sidebar.header("RAG settings")
rag_top_k = st.sidebar.slider("Top-k KB chunks", min_value=2, max_value=12, value=settings.rag_top_k, step=1)
rag_alpha = st.sidebar.slider("Hybrid alpha (dense weight)", min_value=0.0, max_value=1.0, value=float(settings.rag_hybrid_alpha), step=0.05)
st.sidebar.write("If you built embeddings index, queries will use dense+BM25 hybrid. If not, BM25 only.")

# Patch settings object in-memory
settings.openai_api_key = api_key or None
settings.openai_model_fast = model_fast
settings.openai_model_quality = model_quality
settings.rag_top_k = rag_top_k
settings.rag_hybrid_alpha = rag_alpha

# ---------- Main: Spec builder ----------
st.subheader("1) Choose architecture and components")

arch = st.selectbox("Architecture", ["SPN", "FEISTEL", "ARX"], index=0)

colA, colB = st.columns(2)

with colA:
    name = st.text_input("Cipher name", value="MyCipherV2")
    seed = st.number_input("Seed (reproducibility)", min_value=0, max_value=2**31-1, value=int(settings.global_seed), step=1)
    # Default rounds based on architecture
    rounds_default = 10 if arch == "SPN" else (12 if arch == "ARX" else 16)
    rounds = st.slider("Rounds", min_value=2, max_value=40, value=rounds_default, step=1)

with colB:
    if arch == "SPN":
        block_bits = 128
        st.info("SPN template is currently fixed to 128-bit blocks (AES-like components).", icon="ℹ️")
        key_bits = st.selectbox("Key size (bits)", [128, 256], index=0)
    elif arch == "ARX":
        block_bits = st.selectbox("Block size (bits)", [64, 128], index=0)
        st.info("ARX ciphers like RC5 use 64-bit blocks, RC6 uses 128-bit.", icon="ℹ️")
        key_bits = st.selectbox("Key size (bits)", [128, 256], index=0)
    else:  # FEISTEL
        block_bits = st.selectbox("Block size (bits)", [64, 128], index=1)
        key_bits = st.selectbox("Key size (bits)", [128, 256], index=0)

# Component dropdowns
components: Dict[str, str] = {}

if arch == "SPN":
    sboxes = registry.list_by_kind("SBOX", arch="SPN")
    perms = registry.list_by_kind("PERM", arch="SPN")
    lins = registry.list_by_kind("LINEAR", arch="SPN")
    kss = registry.list_by_kind("KEY_SCHEDULE", arch="SPN")

    sbox_id = st.selectbox("S-box", [c.component_id for c in sboxes], index=0)
    perm_id = st.selectbox("Permutation", [c.component_id for c in perms], index=0)
    lin_id = st.selectbox("Linear diffusion", [c.component_id for c in lins], index=0)
    ks_id = st.selectbox("Key schedule", [c.component_id for c in kss], index=0)

    components = {"sbox": sbox_id, "perm": perm_id, "linear": lin_id, "key_schedule": ks_id}

elif arch == "ARX":
    # ARX components: modular addition and rotation
    arx_ops = registry.list_by_kind("ARX", arch="ARX")
    kss = registry.list_by_kind("KEY_SCHEDULE", arch="ARX")
    
    # Separate add and rotate operations
    add_ops = [c for c in arx_ops if "add" in c.component_id or "mul" in c.component_id]
    rot_ops = [c for c in arx_ops if "rotate" in c.component_id]
    
    arx_add_id = st.selectbox("ARX addition/multiplication", [c.component_id for c in add_ops], index=0,
                               help="Modular addition (RC5/RC6) or multiplication (IDEA)")
    arx_rot_id = st.selectbox("ARX rotation", [c.component_id for c in rot_ops], index=0,
                               help="Bit rotation amount per word")
    ks_id = st.selectbox("Key schedule", [c.component_id for c in kss], index=0)

    components = {"arx_add": arx_add_id, "arx_rotate": arx_rot_id, "key_schedule": ks_id}

else:  # FEISTEL
    sboxes = registry.list_by_kind("SBOX", arch="FEISTEL")
    perms = [c for c in registry.list_by_kind("PERM", arch="FEISTEL") if c.component_id == "perm.identity"]
    kss = registry.list_by_kind("KEY_SCHEDULE", arch="FEISTEL")

    f_sbox_id = st.selectbox("F-function S-box", [c.component_id for c in sboxes], index=0)
    f_perm_id = st.selectbox("F-function permutation", [c.component_id for c in perms], index=0)
    ks_id = st.selectbox("Key schedule", [c.component_id for c in kss], index=0)

    components = {"f_sbox": f_sbox_id, "f_perm": f_perm_id, "key_schedule": ks_id}

spec = CipherSpec(
    name=name,
    architecture=arch,
    block_size_bits=int(block_bits),
    key_size_bits=int(key_bits),
    rounds=int(rounds),
    components=components,
    seed=int(seed),
)

ok, errs = validate_spec(spec, registry)
if not ok:
    st.error("Spec validation errors:\n- " + "\n- ".join(errs))
else:
    st.success("Spec looks valid.")

st.subheader("2) Evaluate locally (no API cost)")
metrics: Optional[Dict[str, object]] = None
issues: List[str] = []

if st.button("Run local metrics", disabled=not ok):
    with st.spinner("Running metrics (avalanche tests)…") :
        metrics = evaluate_and_score(spec)
        issues = heuristic_issues(metrics)
    st.session_state["metrics"] = metrics
    st.session_state["issues"] = issues

metrics = st.session_state.get("metrics")
issues = st.session_state.get("issues", [])

if metrics:
    col1, col2 = st.columns(2)
    with col1:
        st.json(metrics)
    with col2:
        st.write("Detected issues:")
        if issues:
            st.warning("\n".join(["- " + x for x in issues]))
        else:
            st.success("No obvious issues flagged by heuristics (still not a security claim).")

st.subheader("3) Export cipher as Python code")

if st.button("Generate Python module", disabled=not ok):
    module_code = export_cipher_module(spec)
    st.session_state["module_code"] = module_code

module_code = st.session_state.get("module_code")
if module_code:
    st.code(module_code, language="python")
    st.download_button("Download cipher_module.py", data=module_code, file_name=f"{spec.name}_cipher.py", mime="text/plain")

    if st.button("Save as reproducible run", disabled=not ok):
        run = make_run_dir(settings.runs_dir, spec.name)
        write_json(run.spec_json, spec.model_dump())
        write_text(run.module_py, module_code)
        if metrics:
            write_json(run.metrics_json, metrics)
        st.success(f"Saved run to: {run.run_dir}")

# ---------- Improvements (RAG + OpenAI call) ----------
st.subheader("4) Ask for improvement suggestions (uses OpenAI)")
st.caption("This makes ONE model call per click. Retrieval + metrics are local.")

if st.button("Suggest improvements", disabled=(not ok or not settings.openai_api_key or not metrics)):
    # Load retriever (may error if index missing)
    try:
        retriever = RAGRetriever(settings)
    except Exception as e:
        st.error(f"RAG retriever error: {e}\n\nRun: python scripts/build_kb_index.py")
        st.stop()

    query = f"Improve diffusion and avalanche for a {spec.architecture} block cipher (block={spec.block_size_bits}, rounds={spec.rounds})."
    if issues:
        query += " Issues: " + " ".join(issues)

    with st.spinner("Retrieving KB context…"):
        chunks = retriever.retrieve(query)
        rag_context = retriever.format_for_prompt(chunks)

    with st.spinner("Calling OpenAI for an ImprovementPatch…"):
        patch, raw = suggest_improvements(
            settings=settings,
            spec=spec,
            metrics=metrics,
            issues=issues,
            rag_context=rag_context,
            model=model_for_improve,
        )

    st.session_state["patch"] = patch
    st.session_state["rag_context"] = [c.__dict__ for c in chunks]

patch = st.session_state.get("patch")
if patch:
    st.write("Suggested patch:")
    st.json(patch.model_dump())

    if st.button("Apply patch and re-evaluate"):
        new_spec = spec.model_copy(deep=True)
        if patch.new_rounds is not None:
            new_spec.rounds = int(patch.new_rounds)
        if patch.replace_components:
            new_spec.components.update(patch.replace_components)
        if patch.add_notes:
            new_spec.notes = (new_spec.notes + "\n" + patch.add_notes).strip()

        ok2, errs2 = validate_spec(new_spec, registry)
        if not ok2:
            st.error("Patched spec invalid:\n- " + "\n- ".join(errs2))
        else:
            with st.spinner("Running metrics for patched spec…"):
                m2 = evaluate_and_score(new_spec)
                iss2 = heuristic_issues(m2)
            st.session_state["new_spec"] = new_spec
            st.session_state["new_metrics"] = m2
            st.session_state["new_issues"] = iss2

new_spec = st.session_state.get("new_spec")
new_metrics = st.session_state.get("new_metrics")
new_issues = st.session_state.get("new_issues", [])

if new_spec and new_metrics:
    st.write("Patched spec:")
    st.json(new_spec.model_dump())
    st.write("Patched metrics:")
    st.json(new_metrics)
    if new_issues:
        st.warning("\n".join(["- " + x for x in new_issues]))
    st.write("Export patched cipher:")
    module2 = export_cipher_module(new_spec)
    st.download_button("Download patched cipher_module.py", data=module2, file_name=f"{new_spec.name}_cipher_patched.py", mime="text/plain")

# ---------- RAG chat ----------
st.subheader("5) KB Chat (block ciphers only)")

question = st.text_input("Ask a question about block ciphers / components / tests", value="")
if st.button("Ask", disabled=(not question.strip() or not settings.openai_api_key)):
    try:
        retriever = RAGRetriever(settings)
    except Exception as e:
        st.error(f"RAG retriever error: {e}\n\nRun: python scripts/build_kb_index.py")
        st.stop()

    chunks = retriever.retrieve(question)
    rag_context = retriever.format_for_prompt(chunks)

    system = "You are a block-cipher research assistant. Use the provided KB snippets. If KB is insufficient, say so."
    user = f"Question: {question}\n\nKB snippets:\n{rag_context}\n\nAnswer clearly and concisely."
    provider = OpenAIProvider(api_key=settings.openai_api_key)
    resp = provider.generate_text(model=model_fast, system=system, user=user, temperature=0.2, max_output_tokens=900)
    st.write(resp.text)

    with st.expander("Retrieved KB chunks"):
        for c in chunks:
            st.markdown(f"**{c.title}** — {c.heading} (`{c.source_path}`)\n\nScore: {c.score:.4f}")
            st.code(c.text[:1200])
