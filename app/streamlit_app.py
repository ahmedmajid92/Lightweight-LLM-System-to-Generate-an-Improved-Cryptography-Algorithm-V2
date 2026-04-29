from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
import streamlit.components.v1 as stc

from cipherlab.config import load_settings
from cipherlab.cipher.metrics import evaluate_and_score, evaluate_full, heuristic_issues
from cipherlab.cipher.registry import ComponentRegistry
from cipherlab.cipher.spec import CipherSpec, ImprovementPatch
from cipherlab.cipher.validator import validate_spec
from cipherlab.cipher.exporter import export_cipher_module
from cipherlab.context_logger import build_context_snapshot, build_copilot_context
from cipherlab.evaluation.roundtrip import run_roundtrip_tests
from cipherlab.evaluation.report import EvaluationReport
from cipherlab.evaluation.feedback import parse_evaluation_results, run_feedback_cycle
from cipherlab.evaluation.avalanche import compute_sac
from cipherlab.evaluation.sbox_analysis import analyze_all_sboxes
from cipherlab.evolution.ast_analyzer import detect_mismatches
from cipherlab.evolution.dynamic_loader import evolve_all_mismatches
from cipherlab.rag.retriever import RAGRetriever
from cipherlab.utils.repro import make_run_dir, write_json, write_text
from cipherlab.llm.assistant import suggest_improvements
from cipherlab.llm.openai_provider import OpenAIProvider
from cipherlab.iteration import (
    IterationHistory,
    IterationRecord,
    MetricsSummary,
    extract_metrics_summary,
)


st.set_page_config(page_title="Crypto Cipher Lab v2", layout="wide")

# --- Scroll-back after rerun: scroll to the section that triggered the rerun ---
_scroll_target = st.session_state.pop("_scroll_to", None)
if _scroll_target:
    stc.html(
        f"""<script>
        window.parent.document.addEventListener('DOMContentLoaded', function() {{
            var el = window.parent.document.querySelector('[data-scroll-id="{_scroll_target}"]');
            if (el) el.scrollIntoView({{block: 'start'}});
        }});
        // Fallback: try immediately (DOM may already be loaded)
        var el = window.parent.document.querySelector('[data-scroll-id="{_scroll_target}"]');
        if (el) el.scrollIntoView({{block: 'start'}});
        </script>""",
        height=0,
    )

settings = load_settings()

st.title("Lightweight LLMs for Block Cipher Algorithms (Generation + Iterative Improvement)")
st.caption(
    "LLM-guided generation and iterative improvement of lightweight block ciphers "
    "for IoT and resource-constrained environments. "
    "Human-in-the-loop refinement backed by deterministic evaluation. "
    "Research prototype — not a claim of cryptographic security."
)

registry = ComponentRegistry()


def _rerun(scroll_to: str = "") -> None:
    """Rerun the app, optionally scrolling back to a section."""
    if scroll_to:
        st.session_state["_scroll_to"] = scroll_to
    st.rerun()


# ===== Helper: display-friendly model label =====

def _model_label(model_id: str) -> str:
    """Map a raw model ID to a generic UI label."""
    mid = (model_id or "").lower()
    if "deepseek" in mid and ("r1" in mid or "reasoning" in mid):
        return "Reasoning model"
    if "deepseek" in mid:
        return "Fast model (primary)"
    if "codex" in mid:
        return "Code model (fallback)"
    if "mini" in mid:
        return "Fast model (fallback)"
    return "Quality model (fallback)"


# ===== Helper: apply patch to spec =====

def apply_patch(base: CipherSpec, patch: ImprovementPatch) -> CipherSpec:
    """Apply an ImprovementPatch to a CipherSpec, returning a new copy."""
    new = base.model_copy(deep=True)
    if patch.new_rounds is not None:
        new.rounds = int(patch.new_rounds)
    if patch.replace_components:
        new.components.update(patch.replace_components)
    if patch.add_notes:
        new.notes = (new.notes + "\n" + patch.add_notes).strip()
    return new


# ===== Helper: build flat metric comparison table =====

def render_metric_comparison(before: MetricsSummary, after: MetricsSummary):
    """Render a before/after metric comparison table."""
    deltas = after.delta(before)
    rows = []
    labels = {
        "pt_avalanche_mean": "PT Avalanche Mean",
        "key_avalanche_mean": "Key Avalanche Mean",
        "pt_avalanche_score": "PT Avalanche Score",
        "key_avalanche_score": "Key Avalanche Score",
        "overall_score": "Overall Score",
        "sac_deviation_pt": "SAC Deviation (PT)",
        "sac_deviation_key": "SAC Deviation (Key)",
    }
    for key, label in labels.items():
        bv = getattr(before, key)
        av = getattr(after, key)
        dv = deltas.get(key)
        if bv is not None or av is not None:
            bv_str = f"{bv:.4f}" if bv is not None else "N/A"
            av_str = f"{av:.4f}" if av is not None else "N/A"
            if dv is not None:
                sign = "+" if dv >= 0 else ""
                dv_str = f"{sign}{dv:.4f}"
            else:
                dv_str = "—"
            rows.append({"Metric": label, "Before": bv_str, "After": av_str, "Delta": dv_str})
    if rows:
        st.table(rows)


# ===== Helper: parse PATCH_PROPOSAL from LLM response =====

_PATCH_MARKER = "PATCH_PROPOSAL:"


def _extract_balanced_json(text: str, start: int) -> Optional[str]:
    """Find the complete JSON object starting at `start` using brace counting."""
    open_pos = text.find("{", start)
    if open_pos == -1:
        return None
    depth = 0
    in_string = False
    escape_next = False
    for i in range(open_pos, len(text)):
        ch = text[i]
        if escape_next:
            escape_next = False
            continue
        if ch == "\\":
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[open_pos : i + 1]
    return None


def parse_patch_proposal(text: str) -> Tuple[Optional[ImprovementPatch], str]:
    """Extract a PATCH_PROPOSAL JSON from an LLM response.

    Handles nested JSON objects (e.g. replace_components: {"sbox": "sbox.aes"}).
    Returns (patch_or_None, display_text_with_json_stripped).
    """
    idx = text.find(_PATCH_MARKER)
    if idx == -1:
        return None, text

    raw_json = _extract_balanced_json(text, idx + len(_PATCH_MARKER))
    if raw_json is None:
        return None, text

    try:
        data = json.loads(raw_json)
        patch = ImprovementPatch.model_validate(data)
    except Exception:
        return None, text

    # Find the full extent of the proposal block (marker + JSON) to strip it
    json_end = text.index(raw_json) + len(raw_json)
    before = text[:idx].rstrip()
    after = text[json_end:].lstrip()
    # Strip surrounding code fences if present
    before = re.sub(r"```\s*$", "", before).rstrip()
    after = re.sub(r"^\s*```", "", after).lstrip()
    display = (before + "\n\n" + after).strip()
    return patch, display


# ========================================================================
# SIDEBAR
# ========================================================================

st.sidebar.header("Model configuration")

# Model IDs come from settings / .env — not exposed as raw text in the UI
model_fast = settings.openai_model_fast
model_quality = settings.openai_model_quality

if settings.openrouter_api_key:
    openrouter_model_choice = st.sidebar.selectbox(
        "Primary model",
        [settings.openrouter_model_fast, settings.openrouter_model_reasoning],
        format_func=lambda m: "Fast model" if "v3" in m else "Reasoning model",
        index=1,  # Default to reasoning as primary model
    )
    st.sidebar.success("Primary reasoning API configured.")
else:
    openrouter_model_choice = None
    st.sidebar.warning("Primary reasoning key not set — improvements will use fallback model.")

model_for_improve = st.sidebar.selectbox(
    "Fallback model",
    ["Quality model", "Fast model"],
    index=0,
)
# Map the display label back to the actual model ID
model_for_improve = model_quality if model_for_improve == "Quality model" else model_fast

st.sidebar.header("RAG settings")
rag_top_k = st.sidebar.slider("Top-k KB chunks", min_value=2, max_value=12, value=settings.rag_top_k, step=1)
rag_alpha = st.sidebar.slider("Hybrid alpha (dense weight)", min_value=0.0, max_value=1.0, value=float(settings.rag_hybrid_alpha), step=0.05)
st.sidebar.write("If you built embeddings index, queries will use dense+BM25 hybrid. If not, BM25 only.")

_kb_index_dir = Path(settings.project_root) / settings.kb_index_dir
_dense_ids_path = _kb_index_dir / "dense_ids.json"
_dense_vecs_path = _kb_index_dir / "embeddings.npy"
_has_dense_index = _dense_ids_path.exists() and _dense_vecs_path.exists()
_can_embed_queries = bool(settings.openai_api_key)
_hybrid_active = settings.rag_use_embeddings and _has_dense_index and _can_embed_queries

if _hybrid_active:
    st.sidebar.success("RAG status: Hybrid retrieval active (BM25 + dense embeddings).")
elif settings.rag_use_embeddings and _has_dense_index and not _can_embed_queries:
    st.sidebar.warning("RAG status: Dense index found, but OPENAI_API_KEY is missing. Using BM25 only.")
elif settings.rag_use_embeddings and not _has_dense_index:
    st.sidebar.info("RAG status: BM25 only. Dense index files were not found in kb_index/.")
else:
    st.sidebar.info("RAG status: BM25 only. Embedding retrieval is disabled by settings.")

# Patch settings object in-memory
settings.rag_top_k = rag_top_k
settings.rag_hybrid_alpha = rag_alpha


# ========================================================================
# SECTION 1 — Choose Architecture and Components
# ========================================================================

st.subheader("1) Choose architecture and components")

arch = st.selectbox("Architecture", ["SPN", "FEISTEL", "ARX"], index=0)

colA, colB = st.columns(2)

with colA:
    name = st.text_input("Cipher name", value="MyCipherV2")
    seed = st.number_input("Seed (reproducibility)", min_value=0, max_value=2**31-1, value=int(settings.global_seed), step=1)
    rounds_default = 10 if arch == "SPN" else (12 if arch == "ARX" else 16)
    rounds = st.slider("Rounds", min_value=2, max_value=64, value=rounds_default, step=1)

with colB:
    if arch == "SPN":
        block_bits = st.selectbox("Block size (bits)", [64, 128], index=1)
        st.info("SPN ciphers: AES (128-bit), PRESENT (64-bit), GIFT (128-bit).")
        key_bits = st.selectbox("Key size (bits)", [80, 128, 256], index=1)
    elif arch == "ARX":
        block_bits = st.selectbox("Block size (bits)", [64, 128], index=0)
        st.info("ARX ciphers: SPECK (64-bit), RC5 (64-bit), LEA (128-bit).")
        key_bits = st.selectbox("Key size (bits)", [128, 256], index=0)
    else:  # FEISTEL
        block_bits = st.selectbox("Block size (bits)", [64, 128], index=1)
        key_bits = st.selectbox("Key size (bits)", [128, 256], index=0)

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
    arx_ops = registry.list_by_kind("ARX", arch="ARX")
    kss = registry.list_by_kind("KEY_SCHEDULE", arch="ARX")

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


# ========================================================================
# WORKING SPEC MANAGEMENT
# ========================================================================

# Detect if the base spec changed (user edited UI dropdowns) → reset iteration state
_base_spec_key = spec.model_dump_json(exclude={"notes", "version"})
if st.session_state.get("_base_spec_key") != _base_spec_key:
    st.session_state["_base_spec_key"] = _base_spec_key
    # Reset iteration-related state
    for _k in ("iteration_history", "pending_patch", "staged_record",
               "metrics", "issues", "eval_report", "eval_diagnostics",
               "feedback_result", "module_code"):
        st.session_state.pop(_k, None)

if "iteration_history" not in st.session_state:
    st.session_state["iteration_history"] = IterationHistory(cipher_name=spec.name)

history: IterationHistory = st.session_state["iteration_history"]

# Working spec = last accepted iteration's result, or the base spec
_last_accepted_dict = history.current_spec_dict()
working_spec: CipherSpec = CipherSpec(**_last_accepted_dict) if _last_accepted_dict else spec


# ========================================================================
# SECTION 2 — Evaluate Locally (No API Cost)
# ========================================================================

st.subheader("2) Evaluate locally")
st.caption("Avalanche tests on the current working design. No model calls required.")

metrics: Optional[Dict[str, object]] = None
issues: List[str] = []

if st.button("Run local metrics", disabled=not ok):
    with st.spinner("Running avalanche tests on working design..."):
        metrics = evaluate_and_score(working_spec, registry=registry)
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


# ========================================================================
# SECTION 3 — Advanced Evaluation (SAC + S-box + I/O Compatibility)
# ========================================================================

st.subheader("3) Advanced evaluation")

with st.expander("Roundtrip verification, SAC analysis, S-box profiling, I/O compatibility", expanded=False):
    st.caption(
        "Deterministic evaluation: roundtrip correctness, Strict Avalanche Criterion, "
        "S-box differential/linear analysis, and component I/O compatibility."
    )

    adv_col1, adv_col2 = st.columns(2)
    with adv_col1:
        adv_num_vectors = st.number_input("Roundtrip test vectors", min_value=100, max_value=10000, value=1000, step=100)
    with adv_col2:
        adv_sac_trials = st.number_input("SAC trials per bit", min_value=50, max_value=2000, value=500, step=50)

    if st.button("Run full evaluation", disabled=not ok, key="btn_full_eval"):
        eval_report = EvaluationReport()

        with st.spinner(f"Running roundtrip tests ({adv_num_vectors} vectors)..."):
            rt_result = run_roundtrip_tests(working_spec, num_vectors=int(adv_num_vectors), seed=int(seed))
            eval_report.roundtrip_results = [rt_result]

        from cipherlab.cipher.builder import build_cipher as pkg_build_cipher
        cipher_obj = pkg_build_cipher(working_spec)

        with st.spinner(f"Computing SAC (plaintext, {adv_sac_trials} trials/bit)..."):
            sac_pt = compute_sac(
                cipher_obj,
                block_size_bits=working_spec.block_size_bits,
                key_size_bits=working_spec.key_size_bits,
                input_type="plaintext",
                trials=int(adv_sac_trials),
                seed=int(seed),
                algorithm_name=working_spec.name,
                architecture=working_spec.architecture,
            )

        with st.spinner(f"Computing SAC (key, {adv_sac_trials} trials/bit)..."):
            sac_key = compute_sac(
                cipher_obj,
                block_size_bits=working_spec.block_size_bits,
                key_size_bits=working_spec.key_size_bits,
                input_type="key",
                trials=int(adv_sac_trials),
                seed=int(seed),
                algorithm_name=working_spec.name,
                architecture=working_spec.architecture,
            )

        eval_report.sac_results = [sac_pt, sac_key]

        with st.spinner("Analyzing S-box properties (DDT/LAT)..."):
            sbox_results = analyze_all_sboxes(registry)
            eval_report.sbox_results = sbox_results

        st.session_state["eval_report"] = eval_report

    eval_report = st.session_state.get("eval_report")
    if eval_report:
        for rt in eval_report.roundtrip_results:
            if rt.is_perfect:
                st.success(rt.summary())
            else:
                st.error(rt.summary())

        for sac in eval_report.sac_results:
            if sac.passes_sac:
                st.success(sac.summary())
            else:
                st.warning(sac.summary())

        if eval_report.sbox_results:
            st.write("**S-box Analysis:**")
            for sb in eval_report.sbox_results:
                st.text(sb.summary())

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

        # Feedback synthesis
        has_api = bool(settings.openai_api_key or settings.openrouter_api_key)
        if st.button("Generate AI feedback", disabled=not has_api, key="btn_feedback"):
            rag_ctx = ""
            try:
                retriever = RAGRetriever(settings)
                query = f"Improve {working_spec.architecture} lightweight block cipher (block={working_spec.block_size_bits}, rounds={working_spec.rounds})"
                chunks = retriever.retrieve(query)
                rag_ctx = retriever.format_for_prompt(chunks)
            except Exception:
                pass

            spinner_msg = "Calling reasoning model..." if settings.openrouter_api_key else "Calling fallback model..."
            with st.spinner(spinner_msg):
                fb_result = run_feedback_cycle(settings, working_spec, eval_report, rag_context=rag_ctx)

            st.session_state["feedback_result"] = fb_result

        fb_result = st.session_state.get("feedback_result")
        if fb_result and fb_result.patch:
            st.write(f"**Model used:** {_model_label(fb_result.model_used)}")
            if fb_result.reasoning_trace:
                with st.expander("Reasoning trace"):
                    st.text(fb_result.reasoning_trace[:3000])
            st.write("**Suggested patch:**")
            st.json(fb_result.patch.model_dump())

    # I/O Compatibility Check
    st.divider()
    st.write("**I/O Compatibility Analysis**")
    if st.button("Check component I/O compatibility", key="btn_io_compat"):
        io_mismatches = detect_mismatches(working_spec, registry)
        if io_mismatches:
            for mm in io_mismatches:
                if mm.severity == "blocking":
                    st.error(mm.summary())
                else:
                    st.warning(mm.summary())
            st.session_state["io_mismatches"] = io_mismatches
        else:
            st.success("All components are I/O compatible with current spec.")
            st.session_state["io_mismatches"] = []


# ========================================================================
# SECTION 4 — Iterative Improvement Loop
# ========================================================================

st.markdown('<div data-scroll-id="section-4"></div>', unsafe_allow_html=True)
st.subheader("4) Iterative improvement")
st.caption(
    "Closed-loop refinement: generate an improvement patch, preview it, apply to a staged design, "
    "evaluate before/after metrics, then accept or reject. Each accepted iteration builds on the last."
)

# Show working spec status
_n_accepted = len(history.accepted())
if _n_accepted > 0:
    st.info(
        f"Working design: **{working_spec.name}** — "
        f"{working_spec.architecture}, {working_spec.block_size_bits}b block, "
        f"{working_spec.rounds} rounds — "
        f"after **{_n_accepted}** accepted improvement(s)"
    )
else:
    st.info(
        f"Working design: **{working_spec.name}** — "
        f"{working_spec.architecture}, {working_spec.block_size_bits}b block, "
        f"{working_spec.rounds} rounds — "
        f"base design (no improvements accepted yet)"
    )

# ---------- 4a: Generate improvement patch ----------

has_api = bool(settings.openai_api_key or settings.openrouter_api_key)
has_metrics = metrics is not None
can_generate = ok and has_api and has_metrics
no_staged = st.session_state.get("staged_record") is None

if not has_metrics:
    st.warning("Run local metrics (section 2) before generating improvements.")

if st.button(
    "Generate improvement patch",
    disabled=not (can_generate and no_staged),
    key="btn_gen_patch",
    help="Requires: valid spec, API key, local metrics. Cannot generate while a staged patch is pending review.",
):
    # Retrieve KB context
    rag_ctx = ""
    try:
        retriever = RAGRetriever(settings)
        query = (
            f"Improve diffusion and avalanche for a {working_spec.architecture} "
            f"lightweight block cipher (block={working_spec.block_size_bits}, rounds={working_spec.rounds})."
        )
        if issues:
            query += " Issues: " + " ".join(issues)
        chunks = retriever.retrieve(query)
        rag_ctx = retriever.format_for_prompt(chunks)
    except Exception:
        pass

    spinner_msg = (
        "Generating improvement via reasoning model..."
        if openrouter_model_choice
        else "Generating improvement via fallback model..."
    )
    with st.spinner(spinner_msg):
        patch, raw, actual_model = suggest_improvements(
            settings=settings,
            spec=working_spec,
            metrics=metrics,
            issues=issues,
            rag_context=rag_ctx,
            model=model_for_improve,
            openrouter_model=openrouter_model_choice,
        )

    # Try to extract reasoning trace if the primary model returned one.
    reasoning_trace = None
    try:
        if hasattr(raw, "choices") and raw.choices:
            content = raw.choices[0].message.content or ""
            if "<think>" in content and "</think>" in content:
                start = content.index("<think>") + len("<think>")
                end = content.index("</think>")
                reasoning_trace = content[start:end].strip()
    except Exception:
        pass

    st.session_state["pending_patch"] = {
        "patch": patch,
        "model_used": actual_model,
        "reasoning_trace": reasoning_trace,
    }
    _rerun("section-4")

# ---------- 4b: Preview pending patch ----------

pending = st.session_state.get("pending_patch")
if pending and st.session_state.get("staged_record") is None:
    patch_obj: ImprovementPatch = pending["patch"]

    st.write("---")
    st.write("**Proposed improvement patch:**")
    st.write(f"**Summary:** {patch_obj.summary}")
    if patch_obj.rationale:
        st.write("**Rationale:**")
        for r in patch_obj.rationale:
            st.write(f"- {r}")

    # Show specific changes
    change_cols = st.columns(2)
    with change_cols[0]:
        if patch_obj.new_rounds is not None:
            st.write(f"Rounds: {working_spec.rounds} → **{patch_obj.new_rounds}**")
        if patch_obj.replace_components:
            st.write("**Component changes:**")
            for role, new_id in patch_obj.replace_components.items():
                old_id = working_spec.components.get(role, "(none)")
                st.write(f"- `{role}`: `{old_id}` → `{new_id}`")
    with change_cols[1]:
        st.write(f"**Model:** {_model_label(pending['model_used'])}")
        if pending.get("reasoning_trace"):
            with st.expander("Reasoning trace"):
                st.text(pending["reasoning_trace"][:3000])

    # Apply & evaluate button
    apply_cols = st.columns([3, 1, 1])
    with apply_cols[0]:
        if st.button("Apply & evaluate staged design", key="btn_apply_eval"):
            patched_spec = apply_patch(working_spec, patch_obj)

            # Check I/O mismatches
            mismatches = detect_mismatches(patched_spec, registry)
            blocking = [m for m in mismatches if m.severity == "blocking"]
            evo_note = ""
            if blocking:
                if has_api:
                    with st.spinner("Resolving I/O mismatches via adaptive evolution..."):
                        evo_report = evolve_all_mismatches(settings, patched_spec, registry)
                    if evo_report.all_resolved():
                        evo_note = f"Evolved {evo_report.evolutions_succeeded} component(s) to resolve I/O mismatches."
                    else:
                        evo_note = f"Could not resolve all mismatches: {evo_report.failed_components}"

            # Validate patched spec
            ok2, errs2 = validate_spec(patched_spec, registry)

            # Evaluate BEFORE (working_spec) and AFTER (patched_spec)
            before_summary = extract_metrics_summary(
                metrics=metrics,
                eval_report=st.session_state.get("eval_report"),
                issues=issues,
            )

            after_metrics_raw = None
            after_issues_raw: List[str] = []
            after_summary = MetricsSummary()

            if ok2:
                with st.spinner("Evaluating patched design..."):
                    after_metrics_raw = evaluate_and_score(patched_spec, registry=registry)
                    after_issues_raw = heuristic_issues(after_metrics_raw)
                after_summary = extract_metrics_summary(
                    metrics=after_metrics_raw,
                    issues=after_issues_raw,
                )

            # Build iteration record
            record = IterationRecord(
                iteration_id=history.next_id,
                before_spec=working_spec.model_dump(),
                after_spec=patched_spec.model_dump() if ok2 else None,
                patch=patch_obj.model_dump(),
                patch_summary=patch_obj.summary[:120],
                before_metrics=before_summary,
                after_metrics=after_summary if ok2 else None,
                validation_ok=ok2,
                validation_errors=errs2 if not ok2 else ([evo_note] if evo_note else []),
                model_used=pending["model_used"],
                reasoning_trace=(pending.get("reasoning_trace") or "")[:2000] or None,
                seed=int(seed),
            )
            if ok2:
                record.compute_deltas()

            st.session_state["staged_record"] = record
            _rerun("section-4")

    with apply_cols[1]:
        if st.button("Discard patch", key="btn_discard_patch"):
            st.session_state.pop("pending_patch", None)
            _rerun("section-4")

# ---------- 4c: Staged record — accept or reject ----------

staged: Optional[IterationRecord] = st.session_state.get("staged_record")
if staged is not None:
    st.write("---")
    st.write(f"### Staged improvement #{staged.iteration_id}")
    st.write(f"**Patch:** {staged.patch_summary}")
    st.write(f"**Model:** {_model_label(staged.model_used)}")

    if not staged.validation_ok:
        st.error("Patched spec failed validation:\n- " + "\n- ".join(staged.validation_errors))
        st.warning("You can only reject this patch (validation failed).")
    else:
        if staged.validation_errors:
            for note in staged.validation_errors:
                st.info(note)

        # Before/after comparison
        if staged.before_metrics and staged.after_metrics:
            st.write("**Before / after metric comparison:**")
            render_metric_comparison(staged.before_metrics, staged.after_metrics)

            # Quick verdict
            delta_overall = (staged.metric_deltas or {}).get("overall_score")
            if delta_overall is not None:
                if delta_overall > 0.01:
                    st.success(f"Overall score improved by {delta_overall:+.4f}")
                elif delta_overall < -0.01:
                    st.warning(f"Overall score decreased by {delta_overall:+.4f}")
                else:
                    st.info("Overall score unchanged (within +/-0.01)")

    # Decision input
    decision_reason = st.text_input(
        "Reason for your decision (required for thesis traceability):",
        key="decision_reason_input",
        placeholder="e.g., 'Avalanche improved significantly' or 'Rounds increase not justified by marginal gain'",
    )

    dec_col1, dec_col2, dec_col3 = st.columns([1, 1, 2])
    with dec_col1:
        can_accept = staged.validation_ok is True
        if st.button("Accept", disabled=not can_accept, key="btn_accept", type="primary"):
            staged.status = "accepted"
            staged.decision_reason = decision_reason or "(no reason provided)"
            history.add(staged)
            # Clear stale state from previous spec
            st.session_state.pop("staged_record", None)
            st.session_state.pop("pending_patch", None)
            st.session_state.pop("eval_report", None)
            st.session_state.pop("eval_diagnostics", None)
            st.session_state.pop("feedback_result", None)
            st.session_state.pop("module_code", None)
            # Auto-recompute metrics for the new working spec so the next
            # iteration's Generate button is immediately available.
            if staged.after_spec:
                try:
                    _new_spec = CipherSpec(**staged.after_spec)
                    _new_metrics = evaluate_and_score(_new_spec, registry=registry)
                    _new_issues = heuristic_issues(_new_metrics)
                    st.session_state["metrics"] = _new_metrics
                    st.session_state["issues"] = _new_issues
                except Exception:
                    st.session_state.pop("metrics", None)
                    st.session_state.pop("issues", None)
            else:
                st.session_state.pop("metrics", None)
                st.session_state.pop("issues", None)
            _rerun("section-4")

    with dec_col2:
        if st.button("Reject", key="btn_reject"):
            staged.status = "rejected"
            staged.decision_reason = decision_reason or "(no reason provided)"
            history.add(staged)
            st.session_state.pop("staged_record", None)
            st.session_state.pop("pending_patch", None)
            _rerun("section-4")

# ---------- 4d: Iteration history ----------

if history.count > 0:
    st.write("---")
    st.write("**Iteration history**")

    hist_rows = []
    for rec in history.records:
        status_map = {"accepted": "Accepted", "rejected": "Rejected", "pending": "Pending"}
        delta_str = ""
        if rec.metric_deltas:
            d = rec.metric_deltas.get("overall_score")
            if d is not None:
                delta_str = f"{d:+.4f}"
        hist_rows.append({
            "#": rec.iteration_id,
            "Status": status_map.get(rec.status, rec.status),
            "Patch": rec.patch_summary[:60],
            "Model": _model_label(rec.model_used),
            "Score Delta": delta_str,
            "Reason": rec.decision_reason[:50],
        })
    st.dataframe(hist_rows, use_container_width=True, hide_index=True)

    # Expandable detail for each record
    with st.expander("Detailed iteration records"):
        for rec in history.records:
            st.write(f"**Iteration #{rec.iteration_id}** — {rec.status.upper()}")
            st.write(f"Timestamp: {rec.timestamp}")
            st.write(f"Patch: {rec.patch_summary}")
            st.write(f"Model: {_model_label(rec.model_used)}")
            st.write(f"Reason: {rec.decision_reason}")
            if rec.metric_deltas:
                notable = {k: v for k, v in rec.metric_deltas.items() if v is not None and abs(v) > 0.0005}
                if notable:
                    st.write("Metric deltas: " + ", ".join(f"{k}={v:+.4f}" for k, v in notable.items()))
            if rec.patch:
                st.json(rec.patch)
            st.write("---")

# ---------- 4e: Rollback ----------

accepted_iters = history.accepted()
if len(accepted_iters) > 1:
    st.write("**Rollback to a prior accepted design:**")
    rollback_options = {
        "Original base design": -1,
    }
    for rec in accepted_iters:
        rollback_options[f"Iteration #{rec.iteration_id}: {rec.patch_summary[:50]}"] = rec.iteration_id

    rollback_choice = st.selectbox(
        "Select a checkpoint to rollback to:",
        list(rollback_options.keys()),
        index=len(rollback_options) - 1,
        key="rollback_select",
    )
    rollback_id = rollback_options[rollback_choice]

    if st.button("Rollback", key="btn_rollback"):
        if rollback_id < 0:
            # Revert to original: clear all accepted iterations by resetting history
            st.session_state["iteration_history"] = IterationHistory(cipher_name=spec.name)
        else:
            # Keep only iterations up to and including the selected one
            new_records = [r for r in history.records if r.iteration_id <= rollback_id]
            history.records = new_records
        # Clear stale state
        for _k in ("pending_patch", "staged_record", "metrics", "issues",
                    "eval_report", "eval_diagnostics", "feedback_result", "module_code"):
            st.session_state.pop(_k, None)
        _rerun("section-4")


# ========================================================================
# SECTION 5 — Export & Thesis Artifacts
# ========================================================================

st.subheader("5) Export")
st.caption(
    "Export the current working design as a standalone Python module, "
    "save a reproducible run with iteration history, and generate "
    "LaTeX tables for thesis publication."
)

if st.button("Generate Python module", disabled=not ok, key="btn_export"):
    module_code = export_cipher_module(working_spec)
    st.session_state["module_code"] = module_code

module_code = st.session_state.get("module_code")
if module_code:
    st.code(module_code, language="python")
    st.download_button(
        "Download cipher_module.py",
        data=module_code,
        file_name=f"{working_spec.name}_cipher.py",
        mime="text/plain",
    )

# --- Save reproducible run ---
st.write("---")
st.write("**Save reproducible run**")
st.caption("Saves spec, module, metrics, iteration history, and LaTeX tables to a timestamped directory.")

if st.button("Save as reproducible run", disabled=not ok, key="btn_save_run"):
    run = make_run_dir(settings.runs_dir, working_spec.name)
    write_json(run.spec_json, working_spec.model_dump())
    if module_code:
        write_text(run.module_py, module_code)
    if metrics:
        write_json(run.metrics_json, metrics)
    # Save iteration history
    if history.count > 0:
        write_json(run.iteration_history_json, history.to_export_dict())
        # Generate LaTeX tables
        from cipherlab.evaluation.iteration_latex import export_iteration_tables
        tex_paths = export_iteration_tables(history, str(run.tables_dir))
        st.success(f"Saved run to: {run.run_dir} ({len(tex_paths)} LaTeX table(s) generated)")
    else:
        st.success(f"Saved run to: {run.run_dir}")

# --- Standalone iteration report export ---
if history.count > 0:
    st.write("---")
    st.write("**Iteration report export**")

    exp_col1, exp_col2 = st.columns(2)

    with exp_col1:
        # JSON export (downloadable)
        history_json_str = json.dumps(history.to_export_dict(), indent=2)
        st.download_button(
            "Download iteration history (JSON)",
            data=history_json_str,
            file_name=f"{working_spec.name}_iteration_history.json",
            mime="application/json",
            key="btn_dl_history_json",
        )

    with exp_col2:
        # LaTeX export (downloadable)
        from cipherlab.evaluation.iteration_latex import (
            iteration_summary_table,
            accepted_metrics_table,
            summary_statistics_table,
        )
        latex_combined = "\n\n".join([
            "% Auto-generated by Crypto Cipher Lab v2",
            "% Iteration history LaTeX tables for thesis inclusion",
            f"% Cipher: {working_spec.name}",
            f"% Iterations: {history.count} total, {len(history.accepted())} accepted",
            "",
            iteration_summary_table(history),
            "",
            accepted_metrics_table(history),
            "",
            summary_statistics_table(history),
        ])
        st.download_button(
            "Download iteration tables (LaTeX)",
            data=latex_combined,
            file_name=f"{working_spec.name}_iteration_tables.tex",
            mime="text/plain",
            key="btn_dl_history_latex",
        )

    # Preview tables (rendered)
    with st.expander("Preview thesis tables"):
        # --- Table 1: Iteration summary ---
        st.write("**Iteration history: LLM-guided improvement trajectory**")
        _preview_rows_1 = []
        for _pr in history.records:
            _pb = _pr.before_metrics.overall_score if _pr.before_metrics else None
            _pa = _pr.after_metrics.overall_score if _pr.after_metrics else None
            _pd = (_pr.metric_deltas or {}).get("overall_score")
            _br = _pr.before_spec.get("rounds", "?") if _pr.before_spec else "?"
            _ar = _pr.after_spec.get("rounds", "?") if _pr.after_spec else "?"
            _preview_rows_1.append({
                "#": _pr.iteration_id,
                "Status": _pr.status.capitalize(),
                "Model": _model_label(_pr.model_used),
                "Rounds": f"{_br} → {_ar}" if _br != _ar else str(_br),
                "Score Before": f"{_pb:.4f}" if _pb is not None else "—",
                "Score After": f"{_pa:.4f}" if _pa is not None else "—",
                "Delta": f"{_pd:+.4f}" if _pd is not None else "—",
                "Decision": (_pr.decision_reason or "—")[:40],
            })
        if _preview_rows_1:
            st.dataframe(_preview_rows_1, use_container_width=True, hide_index=True)
        else:
            st.caption("No iterations yet.")

        st.write("---")

        # --- Table 2: Accepted metric deltas ---
        st.write("**Accepted improvement deltas**")
        _accepted_recs = history.accepted()
        _preview_rows_2 = []
        for _pr in _accepted_recs:
            _d = _pr.metric_deltas or {}
            _preview_rows_2.append({
                "#": _pr.iteration_id,
                "Delta PT Aval.": f"{_d.get('pt_avalanche_score', 0):+.4f}" if _d.get("pt_avalanche_score") is not None else "—",
                "Delta Key Aval.": f"{_d.get('key_avalanche_score', 0):+.4f}" if _d.get("key_avalanche_score") is not None else "—",
                "Delta Overall": f"{_d.get('overall_score', 0):+.4f}" if _d.get("overall_score") is not None else "—",
                "Delta SAC PT": f"{_d.get('sac_deviation_pt', 0):+.4f}" if _d.get("sac_deviation_pt") is not None else "—",
                "Delta SAC Key": f"{_d.get('sac_deviation_key', 0):+.4f}" if _d.get("sac_deviation_key") is not None else "—",
                "Patch": (_pr.patch_summary or "—")[:45],
            })
        if _preview_rows_2:
            st.dataframe(_preview_rows_2, use_container_width=True, hide_index=True)
        else:
            st.caption("No accepted iterations yet.")

        st.write("---")

        # --- Table 3: Summary statistics ---
        st.write("**Iterative improvement summary**")
        _total = history.count
        _n_acc = len(_accepted_recs)
        _n_rej = len(history.rejected())
        _acc_rate = _n_acc / _total if _total > 0 else 0.0
        _cum_delta = sum(
            (r.metric_deltas or {}).get("overall_score", 0) or 0
            for r in _accepted_recs
        )
        _best_delta = max(
            (abs((r.metric_deltas or {}).get("overall_score", 0) or 0) for r in _accepted_recs),
            default=0,
        )
        _models_used = sorted(set(_model_label(r.model_used) for r in history.records if r.model_used))
        _stats_rows = [
            {"Statistic": "Total iterations", "Value": str(_total)},
            {"Statistic": "Accepted", "Value": str(_n_acc)},
            {"Statistic": "Rejected", "Value": str(_n_rej)},
            {"Statistic": "Accept rate", "Value": f"{_acc_rate:.1%}"},
            {"Statistic": "Cumulative delta (overall)", "Value": f"{_cum_delta:+.4f}"},
            {"Statistic": "Best single delta (overall)", "Value": f"{_best_delta:+.4f}"},
            {"Statistic": "Models used", "Value": ", ".join(_models_used) or "—"},
        ]
        st.dataframe(_stats_rows, use_container_width=True, hide_index=True)

        # Raw LaTeX for copy/paste
        with st.expander("Raw LaTeX source"):
            st.code(iteration_summary_table(history), language="latex")
            st.code(accepted_metrics_table(history), language="latex")
            st.code(summary_statistics_table(history), language="latex")


# ========================================================================
# SECTION 6 — Design-Review Copilot
# ========================================================================

st.markdown('<div data-scroll-id="section-6"></div>', unsafe_allow_html=True)
st.subheader("6) Design-review copilot")
st.caption(
    "Discuss the current cipher design, evaluation results, accepted and rejected improvements, "
    "tradeoffs, and next steps. KB evidence is retrieved automatically when relevant. "
    "Powered by reasoning model (primary) or fallback model."
)

# Build available component list for the current architecture
_avail_components: Dict[str, List[str]] = {}
for _kind in ("SBOX", "PERM", "LINEAR", "KEY_SCHEDULE", "ARX"):
    _comps = registry.list_by_kind(_kind, arch=working_spec.architecture)
    if _comps:
        _avail_components[_kind.lower()] = [c.component_id for c in _comps]

_avail_text = "\n".join(
    f"  {kind}: {', '.join(ids)}" for kind, ids in _avail_components.items()
)

COPILOT_SYSTEM_PREAMBLE = (
    "You are a design-review copilot for lightweight block cipher research, "
    "specializing in IoT and resource-constrained cryptography.\n\n"
    "You have access to the researcher's current cipher design, evaluation metrics, "
    "improvement history (with accept/reject decisions and reasons), and diagnostics.\n\n"
    "Your role:\n"
    "- Discuss tradeoffs between design choices (component swaps, round counts).\n"
    "- Explain why specific patches were accepted or rejected, citing metric deltas.\n"
    "- Identify which changes improved or harmed specific metrics (avalanche, SAC, DDT/LAT).\n"
    "- Suggest next experiments or alternative approaches based on the iteration history.\n"
    "- Reference KB evidence about lightweight cipher design principles when relevant.\n"
    "- **When the user asks you to change the design** (swap a component, change rounds, etc.), "
    "output a PATCH_PROPOSAL JSON block so the system can apply it. Format:\n\n"
    '```\nPATCH_PROPOSAL:\n{"summary": "Short description of the change", '
    '"rationale": ["reason 1", "reason 2"], '
    '"new_rounds": null, '
    '"replace_components": {"sbox": "sbox.aes"}, '
    '"add_notes": null}\n```\n\n'
    "Fields:\n"
    "- summary (required): one-line description of the change\n"
    "- rationale (optional): list of design reasons\n"
    "- new_rounds (optional): integer, new round count (null to keep current)\n"
    "- replace_components (optional): dict mapping stage role to new component_id from the available list\n"
    "- add_notes (optional): text to append to design notes\n\n"
    f"Available components for {working_spec.architecture} architecture:\n{_avail_text}\n\n"
    "Current component roles and their stage names in the spec:\n"
    + "\n".join(f"  {role}: {cid}" for role, cid in working_spec.components.items()) + "\n\n"
    "IMPORTANT: Only output PATCH_PROPOSAL when the user explicitly asks for a change. "
    "For discussion-only questions, answer normally without a PATCH_PROPOSAL block.\n"
    "IMPORTANT: Only use component IDs from the available list above.\n\n"
    "Rules:\n"
    "- Do NOT claim cryptographic security. All analysis is based on implemented "
    "deterministic heuristics (avalanche, SAC deviation, S-box DDT/LAT, roundtrip correctness).\n"
    "- Cite specific metric values and iteration numbers when answering.\n"
    "- If the user asks about something not in your context, say so clearly.\n"
    "- Keep answers concise and actionable.\n\n"
)

# Build tiered context (Tier 1 + Tier 2)
copilot_context = build_copilot_context(
    spec=working_spec,
    registry=registry,
    metrics=metrics,
    issues=issues,
    iteration_history=history,
    diagnostics=st.session_state.get("eval_diagnostics"),
    pending_patch=st.session_state.get("pending_patch"),
)

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Render all existing messages in order
for msg in st.session_state["chat_history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------- Inline proposed-patch review (persists across reruns) ----------

_chat_patch_data = st.session_state.get("chat_proposed_patch")
if _chat_patch_data is not None:
    _cp = _chat_patch_data  # shorthand
    _cp_patch: ImprovementPatch = _cp["patch_obj"]
    _sr: Optional[IterationRecord] = _cp.get("staged_record")

    # Auto-apply & evaluate on first render (no extra click needed)
    if _sr is None:
        _patched = apply_patch(working_spec, _cp_patch)
        _mismatch = detect_mismatches(_patched, registry)
        _blocking = [m for m in _mismatch if m.severity == "blocking"]
        _evo_note = ""
        if _blocking and bool(settings.openai_api_key or settings.openrouter_api_key):
            with st.spinner("Resolving I/O mismatches…"):
                _evo_rpt = evolve_all_mismatches(settings, _patched, registry)
            if _evo_rpt.all_resolved():
                _evo_note = f"Evolved {_evo_rpt.evolutions_succeeded} component(s)."
            else:
                _evo_note = f"Could not resolve all mismatches: {_evo_rpt.failed_components}"

        _ok2, _errs2 = validate_spec(_patched, registry)
        _before_sum = extract_metrics_summary(
            metrics=metrics,
            eval_report=st.session_state.get("eval_report"),
            issues=issues,
        )
        _after_sum = MetricsSummary()
        if _ok2:
            with st.spinner("Evaluating proposed change…"):
                _am = evaluate_and_score(_patched, registry=registry)
                _ai = heuristic_issues(_am)
            _after_sum = extract_metrics_summary(metrics=_am, issues=_ai)

        _actual_model = _cp.get("model_used", "copilot-chat")
        _sr = IterationRecord(
            iteration_id=history.next_id,
            before_spec=working_spec.model_dump(),
            after_spec=_patched.model_dump() if _ok2 else None,
            patch=_cp_patch.model_dump(),
            patch_summary=_cp_patch.summary[:120],
            before_metrics=_before_sum,
            after_metrics=_after_sum if _ok2 else None,
            validation_ok=_ok2,
            validation_errors=_errs2 if not _ok2 else ([_evo_note] if _evo_note else []),
            model_used=_actual_model,
            seed=int(seed),
        )
        if _ok2:
            _sr.compute_deltas()
        _cp["staged_record"] = _sr

    # --- Render the proposal card ---
    with st.container(border=True):
        st.subheader("Proposed design change")
        st.write(f"**{_cp_patch.summary}**")

        if _cp_patch.rationale:
            for _r in _cp_patch.rationale:
                st.write(f"- {_r}")

        # Show specific changes
        _change_parts = []
        if _cp_patch.new_rounds is not None:
            _change_parts.append(f"Rounds: {working_spec.rounds} → **{_cp_patch.new_rounds}**")
        if _cp_patch.replace_components:
            for _role, _new_id in _cp_patch.replace_components.items():
                _old_id = working_spec.components.get(_role, "(none)")
                _change_parts.append(f"`{_role}`: `{_old_id}` → `{_new_id}`")
        if _change_parts:
            st.write("**Changes:** " + " | ".join(_change_parts))

        # Validation / metrics
        if not _sr.validation_ok:
            st.error("Patched spec failed validation:\n- " + "\n- ".join(_sr.validation_errors))
        elif _sr.before_metrics and _sr.after_metrics:
            st.write("**Before / after metric comparison:**")
            render_metric_comparison(_sr.before_metrics, _sr.after_metrics)
            _d_overall = (_sr.metric_deltas or {}).get("overall_score")
            if _d_overall is not None:
                if _d_overall > 0.01:
                    st.success(f"Overall score improved by {_d_overall:+.4f}")
                elif _d_overall < -0.01:
                    st.warning(f"Overall score decreased by {_d_overall:+.4f}")
                else:
                    st.info("Overall score unchanged (within +/-0.01)")

        # Decision
        _chat_reason = st.text_input(
            "Reason for your decision:",
            key="chat_patch_reason",
            placeholder="e.g., 'Better avalanche' or 'Not justified'",
        )
        _cc1, _cc2, _cc3 = st.columns([1, 1, 2])
        with _cc1:
            _can_accept_chat = _sr.validation_ok is True
            if st.button("Accept", disabled=not _can_accept_chat, key="btn_chat_accept", type="primary"):
                _sr.status = "accepted"
                _sr.decision_reason = _chat_reason or "(no reason provided)"
                history.add(_sr)
                st.session_state.pop("chat_proposed_patch", None)
                st.session_state.pop("eval_report", None)
                st.session_state.pop("eval_diagnostics", None)
                st.session_state.pop("feedback_result", None)
                st.session_state.pop("module_code", None)
                if _sr.after_spec:
                    try:
                        _ns = CipherSpec(**_sr.after_spec)
                        _nm = evaluate_and_score(_ns, registry=registry)
                        _ni = heuristic_issues(_nm)
                        st.session_state["metrics"] = _nm
                        st.session_state["issues"] = _ni
                    except Exception:
                        st.session_state.pop("metrics", None)
                        st.session_state.pop("issues", None)
                st.session_state["chat_history"].append({
                    "role": "assistant",
                    "content": f"Patch **accepted**: {_sr.patch_summary} (iteration #{_sr.iteration_id})",
                })
                _rerun("section-6")
        with _cc2:
            if st.button("Reject", key="btn_chat_reject"):
                _sr.status = "rejected"
                _sr.decision_reason = _chat_reason or "(no reason provided)"
                history.add(_sr)
                st.session_state.pop("chat_proposed_patch", None)
                st.session_state["chat_history"].append({
                    "role": "assistant",
                    "content": f"Patch **rejected**: {_sr.patch_summary} (iteration #{_sr.iteration_id})",
                })
                _rerun("section-6")
        with _cc3:
            if st.button("Discard", key="btn_chat_discard"):
                st.session_state.pop("chat_proposed_patch", None)
                st.session_state["chat_history"].append({
                    "role": "assistant", "content": "Proposed change discarded.",
                })
                _rerun("section-6")

# ---------- Chat input ----------

_has_api = bool(settings.openai_api_key or settings.openrouter_api_key)
if not _has_api:
    st.info("Configure an API key in the sidebar to use the copilot.")

question = st.chat_input(
    "Ask about the design, metrics, history — or request changes like 'swap sbox to AES'",
    disabled=not _has_api,
)

if question:
    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(question)

    # Retrieve KB context (supporting evidence)
    rag_context = ""
    chunks = []
    try:
        retriever = RAGRetriever(settings)
        chunks = retriever.retrieve(question)
        rag_context = retriever.format_for_prompt(chunks)
    except Exception:
        pass

    # Assemble system prompt: preamble + tiered design context
    system = COPILOT_SYSTEM_PREAMBLE + copilot_context

    # Include recent conversation history (last 6 messages)
    chat_history_recent = st.session_state["chat_history"][-6:]
    history_text = ""
    if chat_history_recent:
        history_text = "\n\nRecent conversation:\n"
        for _m in chat_history_recent:
            _role = "User" if _m["role"] == "user" else "Assistant"
            history_text += f"{_role}: {_m['content'][:300]}\n"

    # Build user prompt
    user_prompt = f"Question: {question}\n"
    if rag_context:
        user_prompt += f"\nKB evidence (use if relevant):\n{rag_context}\n"
    user_prompt += history_text
    user_prompt += "\nAnswer clearly and concisely, citing specific measurements and iteration numbers when relevant."

    # Route to reasoning model via OpenRouter if available, else fallback
    provider = OpenAIProvider(
        api_key=settings.openai_api_key,
        openrouter_api_key=settings.openrouter_api_key,
    )

    with st.chat_message("assistant"):
        with st.spinner("Thinking via reasoning model…" if settings.openrouter_api_key else "Thinking via fallback model…"):
            resp, _actual_chat_model = provider.generate_text_with_fallback(
                openrouter_model=settings.openrouter_model_reasoning if settings.openrouter_api_key else None,
                fallback_model=model_fast,
                system=system,
                user=user_prompt,
                primary_temperature=0.3,
                primary_max_tokens=1500,
                fallback_temperature=0.2,
                fallback_max_output_tokens=1200,
            )
    answer = resp.text
    if _actual_chat_model == settings.openrouter_model_reasoning:
        if "<think>" in answer and "</think>" in answer:
            think_end = answer.index("</think>") + len("</think>")
            answer = answer[think_end:].strip()

    # Parse for PATCH_PROPOSAL
    _proposed_patch, display_answer = parse_patch_proposal(answer)

    with st.chat_message("assistant"):
        st.markdown(display_answer)

    # Persist to history (display version, without raw JSON block)
    st.session_state["chat_history"].append({"role": "user", "content": question})
    st.session_state["chat_history"].append({"role": "assistant", "content": display_answer})

    if chunks:
        st.session_state["_last_kb_chunks"] = chunks

    # If a patch was proposed, stage it for inline review
    if _proposed_patch is not None:
        st.session_state["chat_history"].append({
            "role": "assistant",
            "content": f"I've proposed a design change: **{_proposed_patch.summary}**. "
                       "Evaluating now — see the review panel below.",
        })
        st.session_state["chat_proposed_patch"] = {
            "patch_obj": _proposed_patch,
            "model_used": _actual_chat_model,
            "staged_record": None,  # auto-filled on next rerun
        }
        _rerun("section-6")

# KB chunks from the last query (optional reference)
if st.session_state.get("_last_kb_chunks"):
    with st.expander("Retrieved KB chunks (last query)"):
        for c in st.session_state["_last_kb_chunks"]:
            st.markdown(f"**{c.title}** — {c.heading} (`{c.source_path}`)\n\nScore: {c.score:.4f}")
            st.code(c.text[:1200])

if st.button("Clear chat history", key="btn_clear_chat"):
    st.session_state["chat_history"] = []
    st.session_state.pop("chat_proposed_patch", None)
    st.session_state.pop("_last_kb_chunks", None)
    _rerun("section-6")
