from __future__ import annotations

import json
from typing import Dict, List, Optional

import streamlit as st

from cipherlab.config import load_settings
from cipherlab.cipher.metrics import evaluate_and_score, evaluate_full, heuristic_issues
from cipherlab.cipher.registry import ComponentRegistry
from cipherlab.cipher.spec import CipherSpec
from cipherlab.cipher.validator import validate_spec
from cipherlab.cipher.exporter import export_cipher_module
from cipherlab.context_logger import build_context_snapshot
from cipherlab.evaluation.roundtrip import run_roundtrip_tests
from cipherlab.evaluation.report import EvaluationReport
from cipherlab.evaluation.feedback import parse_evaluation_results, run_feedback_cycle
from cipherlab.evaluation.avalanche import compute_sac
from cipherlab.evaluation.sbox_analysis import analyze_all_sboxes
from cipherlab.rag.retriever import RAGRetriever
from cipherlab.utils.repro import make_run_dir, write_json, write_text
from cipherlab.llm.assistant import suggest_improvements
from cipherlab.llm.openai_provider import OpenAIProvider


st.set_page_config(page_title="Crypto Cipher Lab v2", layout="wide")

settings = load_settings()

st.title("Crypto Cipher Lab v2 — Lightweight Block Cipher Builder")
st.caption("Research-only lab: compose SPN/Feistel/ARX lightweight ciphers for IoT, run local metrics, and iterate improvements with hybrid RAG.")

registry = ComponentRegistry()

# ---------- Sidebar: OpenAI + OpenRouter + RAG settings ----------
st.sidebar.header("OpenAI settings")
api_key = st.sidebar.text_input("OPENAI_API_KEY", value=settings.openai_api_key or "", type="password")
model_fast = st.sidebar.text_input("Fast model", value=settings.openai_model_fast)
model_quality = st.sidebar.text_input("Quality model", value=settings.openai_model_quality)

st.sidebar.header("DeepSeek / OpenRouter")
if settings.openrouter_api_key:
    openrouter_model_choice = st.sidebar.selectbox(
        "DeepSeek model for improvements",
        [settings.openrouter_model_fast, settings.openrouter_model_reasoning],
        format_func=lambda m: "DeepSeek-V3 (fast)" if "v3" in m else "DeepSeek-R1 (reasoning)",
        index=0,
    )
    st.sidebar.success("OpenRouter API key configured.")
else:
    openrouter_model_choice = None
    st.sidebar.warning("OPENROUTER_API_KEY not set — improvements will use OpenAI.")

# Fallback model for improvements when OpenRouter is not configured
model_for_improve = st.sidebar.selectbox("Fallback OpenAI model for improvements", [model_fast, model_quality], index=0)

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
    rounds = st.slider("Rounds", min_value=2, max_value=64, value=rounds_default, step=1)

with colB:
    if arch == "SPN":
        block_bits = st.selectbox("Block size (bits)", [64, 128], index=1)
        st.info("SPN ciphers: AES (128-bit), PRESENT (64-bit), GIFT (128-bit).", icon="ℹ️")
        key_bits = st.selectbox("Key size (bits)", [80, 128, 256], index=1)
    elif arch == "ARX":
        block_bits = st.selectbox("Block size (bits)", [64, 128], index=0)
        st.info("ARX ciphers: SPECK (64-bit), RC5 (64-bit), LEA (128-bit).", icon="ℹ️")
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
                               help="Modular addition (SPECK/RC5/LEA) or multiplication (legacy)")
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

# ---------- Advanced Evaluation (Phase 3) ----------
with st.expander("Advanced evaluation (SAC + S-box analysis)", expanded=False):
    st.caption("Deterministic cryptographic evaluation: roundtrip verification, Strict Avalanche Criterion, and S-box differential/linear analysis.")

    adv_col1, adv_col2 = st.columns(2)
    with adv_col1:
        adv_num_vectors = st.number_input("Roundtrip test vectors", min_value=100, max_value=10000, value=1000, step=100)
    with adv_col2:
        adv_sac_trials = st.number_input("SAC trials per bit", min_value=50, max_value=2000, value=500, step=50)

    if st.button("Run full evaluation", disabled=not ok, key="btn_full_eval"):
        eval_report = EvaluationReport()

        # Step 1: Roundtrip verification
        with st.spinner(f"Running roundtrip tests ({adv_num_vectors} vectors)…"):
            rt_result = run_roundtrip_tests(spec, num_vectors=int(adv_num_vectors), seed=int(seed))
            eval_report.roundtrip_results = [rt_result]

        # Step 2: SAC analysis
        from cipherlab.cipher.builder import build_cipher as pkg_build_cipher
        cipher_obj = pkg_build_cipher(spec)

        with st.spinner(f"Computing SAC (plaintext, {adv_sac_trials} trials/bit)…"):
            sac_pt = compute_sac(
                cipher_obj,
                block_size_bits=spec.block_size_bits,
                key_size_bits=spec.key_size_bits,
                input_type="plaintext",
                trials=int(adv_sac_trials),
                seed=int(seed),
                algorithm_name=spec.name,
                architecture=spec.architecture,
            )

        with st.spinner(f"Computing SAC (key, {adv_sac_trials} trials/bit)…"):
            sac_key = compute_sac(
                cipher_obj,
                block_size_bits=spec.block_size_bits,
                key_size_bits=spec.key_size_bits,
                input_type="key",
                trials=int(adv_sac_trials),
                seed=int(seed),
                algorithm_name=spec.name,
                architecture=spec.architecture,
            )

        eval_report.sac_results = [sac_pt, sac_key]

        # Step 3: S-box analysis
        with st.spinner("Analyzing S-box properties (DDT/LAT)…"):
            sbox_results = analyze_all_sboxes(registry)
            eval_report.sbox_results = sbox_results

        st.session_state["eval_report"] = eval_report

    eval_report = st.session_state.get("eval_report")
    if eval_report:
        # Display roundtrip results
        for rt in eval_report.roundtrip_results:
            if rt.is_perfect:
                st.success(rt.summary())
            else:
                st.error(rt.summary())

        # Display SAC results
        for sac in eval_report.sac_results:
            if sac.passes_sac:
                st.success(sac.summary())
            else:
                st.warning(sac.summary())

        # Display S-box results
        if eval_report.sbox_results:
            st.write("**S-box Analysis:**")
            for sb in eval_report.sbox_results:
                st.text(sb.summary())

        # Parse diagnostics
        diagnostics = parse_evaluation_results(eval_report)
        if diagnostics:
            st.write(f"**{len(diagnostics)} diagnostic(s) found:**")
            for d in diagnostics:
                if d.severity == "critical":
                    st.error(d.to_prompt_block())
                else:
                    st.warning(d.to_prompt_block())
            st.session_state["eval_diagnostics"] = diagnostics
        else:
            st.success("No issues detected in advanced evaluation.")
            st.session_state["eval_diagnostics"] = []

        # Feedback synthesis button
        has_api = bool(settings.openai_api_key or settings.openrouter_api_key)
        if st.button("Generate AI feedback (DeepSeek-R1 / OpenAI)", disabled=not has_api, key="btn_feedback"):
            rag_ctx = ""
            try:
                retriever = RAGRetriever(settings)
                query = f"Improve {spec.architecture} lightweight block cipher (block={spec.block_size_bits}, rounds={spec.rounds})"
                chunks = retriever.retrieve(query)
                rag_ctx = retriever.format_for_prompt(chunks)
            except Exception:
                pass

            spinner_msg = "Calling DeepSeek-R1 via OpenRouter…" if settings.openrouter_api_key else "Calling OpenAI…"
            with st.spinner(spinner_msg):
                fb_result = run_feedback_cycle(settings, spec, eval_report, rag_context=rag_ctx)

            st.session_state["feedback_result"] = fb_result

        fb_result = st.session_state.get("feedback_result")
        if fb_result and fb_result.patch:
            st.write(f"**Model used:** {fb_result.model_used}")
            if fb_result.reasoning_trace:
                with st.expander("DeepSeek-R1 reasoning trace"):
                    st.text(fb_result.reasoning_trace[:3000])
            st.write("**Suggested patch:**")
            st.json(fb_result.patch.model_dump())

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

# ---------- Improvements (RAG + DeepSeek/OpenAI call) ----------
st.subheader("4) Ask for improvement suggestions")
st.caption("This makes ONE model call per click. Retrieval + metrics are local.")

if st.button("Suggest improvements", disabled=(not ok or not settings.openai_api_key or not metrics)):
    # Load retriever (may error if index missing)
    try:
        retriever = RAGRetriever(settings)
    except Exception as e:
        st.error(f"RAG retriever error: {e}\n\nRun: python scripts/build_kb_index.py")
        st.stop()

    query = f"Improve diffusion and avalanche for a {spec.architecture} lightweight block cipher (block={spec.block_size_bits}, rounds={spec.rounds})."
    if issues:
        query += " Issues: " + " ".join(issues)

    with st.spinner("Retrieving KB context…"):
        chunks = retriever.retrieve(query)
        rag_context = retriever.format_for_prompt(chunks)

    spinner_msg = "Calling DeepSeek via OpenRouter…" if openrouter_model_choice else "Calling OpenAI for an ImprovementPatch…"
    with st.spinner(spinner_msg):
        patch, raw = suggest_improvements(
            settings=settings,
            spec=spec,
            metrics=metrics,
            issues=issues,
            rag_context=rag_context,
            model=model_for_improve,
            openrouter_model=openrouter_model_choice,
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
st.subheader("5) KB Chat (lightweight block ciphers)")

# Build context snapshot for the current cipher design
ctx_snapshot = build_context_snapshot(spec, registry, metrics, issues)
cipher_context = ctx_snapshot.to_prompt_context()

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Display existing chat history
for msg in st.session_state["chat_history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.text_input("Ask a question about lightweight block ciphers / components / tests", value="")
if st.button("Ask", disabled=(not question.strip() or not settings.openai_api_key)):
    try:
        retriever = RAGRetriever(settings)
    except Exception as e:
        st.error(f"RAG retriever error: {e}\n\nRun: python scripts/build_kb_index.py")
        st.stop()

    chunks = retriever.retrieve(question)
    rag_context = retriever.format_for_prompt(chunks)

    # Context-aware system prompt
    system = (
        "You are a lightweight block-cipher research assistant focused on IoT and "
        "resource-constrained cryptography. Use the provided KB snippets. If KB is insufficient, say so.\n\n"
        "Current design context:\n" + cipher_context
    )

    # Include recent conversation history (last 6 messages)
    history = st.session_state["chat_history"][-6:]
    history_text = ""
    if history:
        history_text = "\n\nConversation history:\n"
        for msg in history:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_text += f"{role}: {msg['content'][:300]}\n"

    user = f"Question: {question}\n\nKB snippets:\n{rag_context}{history_text}\n\nAnswer clearly and concisely."
    provider = OpenAIProvider(api_key=settings.openai_api_key)
    resp = provider.generate_text(model=model_fast, system=system, user=user, temperature=0.2, max_output_tokens=900)

    # Store in chat history
    st.session_state["chat_history"].append({"role": "user", "content": question})
    st.session_state["chat_history"].append({"role": "assistant", "content": resp.text})

    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        st.write(resp.text)

    with st.expander("Retrieved KB chunks"):
        for c in chunks:
            st.markdown(f"**{c.title}** — {c.heading} (`{c.source_path}`)\n\nScore: {c.score:.4f}")
            st.code(c.text[:1200])

if st.button("Clear chat history"):
    st.session_state["chat_history"] = []
    st.rerun()
