from __future__ import annotations

import html
import inspect

import gradio as gr


STEP_DETAILS = [
    {
        "label": "1. Choose architecture",
        "title": "Choose architecture and components",
        "tagline": "Build the research cipher configuration before any testing starts.",
        "goal": "Create a valid CipherSpec from architecture, sizes, rounds, and registered components.",
        "inputs": [
            "Architecture family: SPN, FEISTEL, or ARX",
            "Cipher name, seed, rounds, block size, key size",
            "Component IDs from the registry",
        ],
        "engine": [
            "Assemble a structured CipherSpec object",
            "Validate component compatibility and required slots",
            "Store the base design as the current working spec",
        ],
        "outputs": [
            "Validated base design",
            "Working spec in session state",
            "Clean starting point for evaluation",
        ],
        "model_use": "No model required",
        "saved_state": "CipherSpec, working spec key, architecture-specific component choices",
        "swatch": "#14532d",
        "accent": "#22c55e",
        "diagram": ["Researcher input", "CipherSpec build", "Validation", "Working design"],
    },
    {
        "label": "2. Evaluate locally",
        "title": "Evaluate locally with no API cost",
        "tagline": "Measure baseline diffusion before asking for any AI help.",
        "goal": "Run deterministic avalanche tests on the current working design.",
        "inputs": [
            "Current working spec",
            "Registry-built cipher implementation",
            "Seeded random test vectors",
        ],
        "engine": [
            "Compute plaintext avalanche mean",
            "Compute key avalanche mean",
            "Convert both values into compact scores",
        ],
        "outputs": [
            "Metrics JSON",
            "Overall avalanche score",
            "Heuristic issue list",
        ],
        "model_use": "No model required",
        "saved_state": "metrics, issues",
        "swatch": "#1d4ed8",
        "accent": "#60a5fa",
        "diagram": ["Working design", "Avalanche tests", "Scores", "Issue flags"],
    },
    {
        "label": "3. Advanced evaluation",
        "title": "Run deeper evaluation and diagnostics",
        "tagline": "Verify correctness, probe SAC behavior, profile S-boxes, and detect interface issues.",
        "goal": "Expand the evaluation surface beyond basic avalanche scores.",
        "inputs": [
            "Current working spec",
            "Roundtrip vector count",
            "SAC trial count per bit",
        ],
        "engine": [
            "Roundtrip verification: P = D(E(P,K),K)",
            "SAC checks for plaintext and key inputs",
            "S-box DDT/LAT profiling and I/O compatibility analysis",
        ],
        "outputs": [
            "EvaluationReport",
            "Structured diagnostics",
            "Optional AI feedback patch",
        ],
        "model_use": "Optional reasoning model or fallback model for feedback synthesis",
        "saved_state": "eval_report, eval_diagnostics, feedback_result, io_mismatches",
        "swatch": "#6b21a8",
        "accent": "#c084fc",
        "diagram": ["Working design", "Roundtrip + SAC", "Diagnostics", "Feedback-ready report"],
    },
    {
        "label": "4. Iterative improvement",
        "title": "Generate, stage, compare, and decide",
        "tagline": "Turn diagnostics into a controlled human-in-the-loop improvement loop.",
        "goal": "Propose small patches, evaluate them, and keep only accepted improvements.",
        "inputs": [
            "Working spec, local metrics, detected issues",
            "Knowledge-base context from retrieval",
            "Human acceptance or rejection decision",
        ],
        "engine": [
            "Ask a reasoning model or fallback model for an ImprovementPatch",
            "Detect and optionally evolve component mismatches",
            "Evaluate before/after summaries and compute metric deltas",
        ],
        "outputs": [
            "Staged patch review",
            "Accepted or rejected iteration record",
            "Rollback-capable history",
        ],
        "model_use": "Reasoning model preferred, quality or fast fallback when needed",
        "saved_state": "pending_patch, staged_record, iteration_history",
        "swatch": "#9a3412",
        "accent": "#fb923c",
        "diagram": ["Metrics + KB", "Patch proposal", "Staged evaluation", "Accept / reject"],
    },
    {
        "label": "5. Thesis artifacts",
        "title": "Export artifacts for reproducibility and writing",
        "tagline": "Package the current design and the full iteration trace for thesis use.",
        "goal": "Move from live experimentation into reusable research outputs.",
        "inputs": [
            "Current working spec",
            "Generated module code",
            "Iteration history and metrics",
        ],
        "engine": [
            "Generate standalone Python cipher module",
            "Save reproducible run folders",
            "Produce JSON and LaTeX thesis tables",
        ],
        "outputs": [
            "cipher_module.py",
            "Saved run directory",
            "JSON + LaTeX export packages",
        ],
        "model_use": "No model required for export itself",
        "saved_state": "module_code, run artifacts on disk, downloadable tables",
        "swatch": "#0f766e",
        "accent": "#2dd4bf",
        "diagram": ["Working design", "Exporter", "Run snapshot", "Thesis outputs"],
    },
    {
        "label": "6. Design-review copilot",
        "title": "Discuss the design with a context-aware copilot",
        "tagline": "Use chat to review tradeoffs, ask questions, or request controlled design changes.",
        "goal": "Provide a conversational layer grounded in the current design, history, diagnostics, and KB evidence.",
        "inputs": [
            "Chat question",
            "Tiered context from spec, metrics, diagnostics, and history",
            "Retrieved knowledge-base snippets",
        ],
        "engine": [
            "Build a focused system prompt with current design context",
            "Route to the reasoning model or fast fallback model",
            "Parse PATCH_PROPOSAL blocks into staged design changes",
        ],
        "outputs": [
            "Concise answer grounded in current state",
            "Optional staged patch card",
            "Updated chat history",
        ],
        "model_use": "Reasoning model preferred, fast fallback model otherwise",
        "saved_state": "chat_history, chat_proposed_patch, last KB chunks",
        "swatch": "#7c2d12",
        "accent": "#f97316",
        "diagram": ["Question", "Tiered context + KB", "Copilot answer", "Optional patch proposal"],
    },
]


STEP_BY_LABEL = {step["label"]: step for step in STEP_DETAILS}


CUSTOM_CSS = """
.gradio-container {
    background:
        radial-gradient(circle at top left, rgba(34, 197, 94, 0.16), transparent 28%),
        radial-gradient(circle at top right, rgba(96, 165, 250, 0.18), transparent 30%),
        linear-gradient(180deg, #f4f8f4 0%, #f9fafb 52%, #eef6ff 100%);
    color: #14231b;
}

.app-shell {
    max-width: 1260px;
    margin: 0 auto;
}

.hero-card,
.section-card,
.diagram-card,
.stats-card,
.artifact-card,
.detail-card {
    background: rgba(255, 255, 255, 0.88);
    border: 1px solid rgba(20, 35, 27, 0.08);
    border-radius: 22px;
    box-shadow: 0 18px 48px rgba(15, 23, 42, 0.08);
}

.hero-card {
    padding: 26px 28px 22px;
}

.hero-grid,
.overview-grid,
.two-col-grid,
.detail-grid,
.artifact-grid,
.kpi-grid {
    display: grid;
    gap: 16px;
}

.hero-grid {
    grid-template-columns: 1.15fr 0.85fr;
}

.overview-grid,
.artifact-grid {
    grid-template-columns: repeat(3, minmax(0, 1fr));
}

.diagram-stack {
    display: grid;
    gap: 18px;
    align-content: start;
}

.detail-grid,
.two-col-grid {
    grid-template-columns: 0.95fr 1.05fr;
}

.kpi-grid {
    grid-template-columns: repeat(3, minmax(0, 1fr));
    margin-top: 16px;
}

.section-card,
.diagram-card,
.artifact-card,
.detail-card,
.stats-card {
    padding: 18px 20px;
}

.eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 12px;
    border-radius: 999px;
    background: rgba(20, 35, 27, 0.06);
    color: #14532d;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.hero-title {
    margin: 12px 0 8px;
    font-size: 34px;
    line-height: 1.12;
    font-weight: 800;
    color: #102117;
}

.hero-subtitle {
    margin: 0;
    font-size: 15px;
    line-height: 1.65;
    color: #334155;
}

.kpi {
    padding: 16px 18px;
    border-radius: 18px;
    background: linear-gradient(180deg, rgba(248, 250, 252, 0.96), rgba(239, 246, 255, 0.78));
    border: 1px solid rgba(20, 35, 27, 0.07);
}

.kpi-value {
    font-size: 28px;
    font-weight: 800;
    color: #0f172a;
}

.kpi-label {
    margin-top: 6px;
    font-size: 13px;
    color: #475569;
}

.section-title {
    margin: 0 0 10px;
    font-size: 22px;
    font-weight: 800;
    color: #102117;
}

.section-copy {
    margin: 0;
    font-size: 14px;
    line-height: 1.68;
    color: #475569;
}

.pill-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 14px;
}

.pill {
    display: inline-flex;
    align-items: center;
    padding: 8px 12px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 700;
    background: #eef7ff;
    border: 1px solid rgba(14, 116, 144, 0.12);
    color: #0f172a;
}

.mini-table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 12px;
    overflow: hidden;
    border-radius: 16px;
}

.mini-table th,
.mini-table td {
    padding: 11px 12px;
    border-bottom: 1px solid rgba(148, 163, 184, 0.18);
    font-size: 13px;
    vertical-align: top;
}

.mini-table th {
    width: 26%;
    text-align: left;
    font-weight: 800;
    color: #102117;
    background: rgba(241, 245, 249, 0.95);
}

.mini-table td {
    color: #334155;
    background: rgba(255, 255, 255, 0.9);
}

.step-strip {
    display: grid;
    grid-template-columns: repeat(6, minmax(0, 1fr));
    gap: 10px;
    margin-bottom: 16px;
}

.step-chip {
    border-radius: 18px;
    padding: 14px 12px;
    color: #ffffff;
    min-height: 84px;
}

.step-chip small {
    display: block;
    opacity: 0.82;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.step-chip strong {
    display: block;
    margin-top: 8px;
    font-size: 14px;
    line-height: 1.3;
}

.flow-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 12px;
    margin-top: 14px;
}

.flow-card {
    border-radius: 18px;
    padding: 14px 15px;
    background: linear-gradient(180deg, rgba(248, 250, 252, 0.95), rgba(241, 245, 249, 0.88));
    border: 1px solid rgba(148, 163, 184, 0.16);
}

.flow-card h4 {
    margin: 0 0 8px;
    font-size: 14px;
    font-weight: 800;
    color: #0f172a;
}

.flow-card ul {
    margin: 0;
    padding-left: 16px;
    color: #475569;
    font-size: 13px;
    line-height: 1.55;
}

.legend-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 14px;
}

.legend-chip {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 12px;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.9);
    border: 1px solid rgba(15, 23, 42, 0.08);
    font-size: 12px;
    font-weight: 700;
    color: #0f172a;
}

.legend-dot {
    width: 10px;
    height: 10px;
    border-radius: 999px;
}

.artifact-card h4 {
    margin: 0 0 8px;
    font-size: 16px;
    font-weight: 800;
}

.artifact-card p {
    margin: 0;
    font-size: 13px;
    line-height: 1.62;
    color: #475569;
}

.tab-note {
    margin-top: 8px;
    font-size: 13px;
    color: #64748b;
}

@media (max-width: 1080px) {
    .hero-grid,
    .overview-grid,
    .detail-grid,
    .two-col-grid,
    .artifact-grid,
    .step-strip,
    .flow-grid,
    .kpi-grid {
        grid-template-columns: 1fr;
    }
}
"""


def _pill(text: str) -> str:
    return f"<span class='pill'>{html.escape(text)}</span>"


def _step_strip() -> str:
    chips = []
    for step in STEP_DETAILS:
        chips.append(
            f"""
            <div class="step-chip" style="background: linear-gradient(135deg, {step['swatch']} 0%, {step['accent']} 100%);">
                <small>{html.escape(step['label'])}</small>
                <strong>{html.escape(step['title'])}</strong>
            </div>
            """
        )
    return "<div class='step-strip'>" + "".join(chips) + "</div>"


def _mini_stage_svg(nodes: list[str], swatch: str, accent: str) -> str:
    box_width = 150
    start_x = 28
    y = 64
    arrow_y = 102
    parts = []
    for idx, label in enumerate(nodes):
        x = start_x + idx * 175
        fill = swatch if idx in (0, 3) else accent
        parts.append(
            f"""
            <rect x="{x}" y="{y}" width="{box_width}" height="76" rx="20" fill="{fill}" opacity="0.95"/>
            <text x="{x + box_width / 2}" y="{y + 28}" fill="#ffffff" font-size="13" font-weight="800" text-anchor="middle">{html.escape(label.split()[0])}</text>
            <text x="{x + box_width / 2}" y="{y + 47}" fill="#ffffff" font-size="11" font-weight="500" text-anchor="middle">{html.escape(" ".join(label.split()[1:]))}</text>
            """
        )
        if idx < len(nodes) - 1:
            parts.append(
                f"""
                <line x1="{x + box_width}" y1="{arrow_y}" x2="{x + 175}" y2="{arrow_y}" stroke="#0f172a" stroke-width="3" marker-end="url(#mini-arrow)"/>
                """
            )
    return f"""
    <svg viewBox="0 0 760 210" width="100%" height="210" role="img" aria-label="Stage mini flow">
        <defs>
            <linearGradient id="mini-bg" x1="0" x2="1">
                <stop offset="0%" stop-color="#f8fafc"/>
                <stop offset="100%" stop-color="#ecfeff"/>
            </linearGradient>
            <marker id="mini-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
                <path d="M 0 0 L 10 5 L 0 10 z" fill="#0f172a"/>
            </marker>
        </defs>
        <rect x="4" y="4" width="752" height="202" rx="24" fill="url(#mini-bg)" stroke="rgba(15, 23, 42, 0.08)"/>
        <text x="28" y="36" fill="#0f172a" font-size="16" font-weight="800">Stage flow</text>
        {''.join(parts)}
    </svg>
    """


def _render_step_detail(step_label: str) -> str:
    step = STEP_BY_LABEL[step_label]
    inputs = "".join(f"<li>{html.escape(item)}</li>" for item in step["inputs"])
    engine = "".join(f"<li>{html.escape(item)}</li>" for item in step["engine"])
    outputs = "".join(f"<li>{html.escape(item)}</li>" for item in step["outputs"])
    stage_svg = _mini_stage_svg(step["diagram"], step["swatch"], step["accent"])
    return f"""
    <div class="detail-card">
        <div class="eyebrow">{html.escape(step['label'])}</div>
        <h2 class="section-title" style="margin-top:12px;">{html.escape(step['title'])}</h2>
        <p class="section-copy">{html.escape(step['tagline'])}</p>
        <div class="detail-grid" style="margin-top:16px;">
            <div>
                <table class="mini-table">
                    <tr><th>Goal</th><td>{html.escape(step['goal'])}</td></tr>
                    <tr><th>Model use</th><td>{html.escape(step['model_use'])}</td></tr>
                    <tr><th>Saved state</th><td>{html.escape(step['saved_state'])}</td></tr>
                </table>
                <div class="flow-grid">
                    <div class="flow-card">
                        <h4>Inputs</h4>
                        <ul>{inputs}</ul>
                    </div>
                    <div class="flow-card">
                        <h4>System work</h4>
                        <ul>{engine}</ul>
                    </div>
                    <div class="flow-card">
                        <h4>Outputs</h4>
                        <ul>{outputs}</ul>
                    </div>
                </div>
            </div>
            <div class="diagram-card">
                {stage_svg}
            </div>
        </div>
    </div>
    """


def _overview_map_svg() -> str:
    return """
    <svg viewBox="0 0 820 520" width="100%" height="420" role="img" aria-label="System architecture">
        <defs>
            <linearGradient id="layer-a" x1="0" x2="1">
                <stop offset="0%" stop-color="#14532d"/>
                <stop offset="100%" stop-color="#22c55e"/>
            </linearGradient>
            <linearGradient id="layer-b" x1="0" x2="1">
                <stop offset="0%" stop-color="#1d4ed8"/>
                <stop offset="100%" stop-color="#60a5fa"/>
            </linearGradient>
            <linearGradient id="layer-c" x1="0" x2="1">
                <stop offset="0%" stop-color="#7c3aed"/>
                <stop offset="100%" stop-color="#c084fc"/>
            </linearGradient>
            <linearGradient id="layer-d" x1="0" x2="1">
                <stop offset="0%" stop-color="#0f766e"/>
                <stop offset="100%" stop-color="#2dd4bf"/>
            </linearGradient>
            <filter id="soft-shadow" x="-20%" y="-20%" width="140%" height="140%">
                <feDropShadow dx="0" dy="12" stdDeviation="14" flood-opacity="0.12"/>
            </filter>
        </defs>
        <rect x="8" y="8" width="804" height="504" rx="30" fill="#ffffff" opacity="0.88"/>

        <rect x="44" y="48" width="732" height="78" rx="24" fill="url(#layer-a)" filter="url(#soft-shadow)"/>
        <text x="68" y="82" fill="#ffffff" font-size="16" font-weight="800">Interaction layer</text>
        <text x="68" y="105" fill="#ecfdf5" font-size="13">Streamlit research app + separate Gradio explainer UI</text>

        <rect x="76" y="158" width="668" height="82" rx="24" fill="url(#layer-b)" filter="url(#soft-shadow)"/>
        <text x="100" y="191" fill="#ffffff" font-size="16" font-weight="800">State and control layer</text>
        <text x="100" y="214" fill="#eff6ff" font-size="13">working spec, metrics, diagnostics, pending patch, chat history, iteration history</text>

        <rect x="108" y="270" width="604" height="88" rx="24" fill="url(#layer-c)" filter="url(#soft-shadow)"/>
        <text x="132" y="305" fill="#ffffff" font-size="16" font-weight="800">Deterministic engine layer</text>
        <text x="132" y="330" fill="#f5f3ff" font-size="13">registry, builder, validation, avalanche, SAC, roundtrip, S-box analysis, mismatch detection</text>

        <rect x="140" y="388" width="540" height="86" rx="24" fill="url(#layer-d)" filter="url(#soft-shadow)"/>
        <text x="164" y="422" fill="#ffffff" font-size="16" font-weight="800">Knowledge and model layer</text>
        <text x="164" y="446" fill="#ecfeff" font-size="13">hybrid retrieval, reasoning model, quality model fallback, fast model fallback, code model for component repair</text>

        <line x1="410" y1="126" x2="410" y2="158" stroke="#0f172a" stroke-width="4" stroke-linecap="round"/>
        <line x1="410" y1="240" x2="410" y2="270" stroke="#0f172a" stroke-width="4" stroke-linecap="round"/>
        <line x1="410" y1="358" x2="410" y2="388" stroke="#0f172a" stroke-width="4" stroke-linecap="round"/>
    </svg>
    """


def _signals_svg() -> str:
    bars = [
        ("Avalanche", 88, "#1d4ed8"),
        ("Roundtrip", 96, "#0f766e"),
        ("SAC", 74, "#7c3aed"),
        ("S-box", 62, "#f97316"),
        ("I/O fit", 68, "#14532d"),
        ("History", 82, "#475569"),
    ]
    parts = []
    start_x = 84
    base_y = 292
    gap = 96
    for idx, (label, value, color) in enumerate(bars):
        x = start_x + idx * gap
        h = value * 1.8
        y = base_y - h
        parts.append(
            f"""
            <rect x="{x}" y="{y}" width="48" height="{h}" rx="14" fill="{color}" opacity="0.92"/>
            <text x="{x + 24}" y="{base_y + 24}" text-anchor="middle" fill="#0f172a" font-size="12" font-weight="700">{html.escape(label)}</text>
            <text x="{x + 24}" y="{y - 10}" text-anchor="middle" fill="#0f172a" font-size="12" font-weight="800">{value}</text>
            """
        )
    return f"""
    <svg viewBox="0 0 760 340" width="100%" height="310" role="img" aria-label="Signal coverage chart">
        <rect x="4" y="4" width="752" height="332" rx="26" fill="#ffffff" opacity="0.9"/>
        <text x="28" y="36" fill="#0f172a" font-size="18" font-weight="800">Signal coverage across the framework</text>
        <text x="28" y="58" fill="#475569" font-size="12">Higher bars mean stronger visibility into the current design state.</text>
        <line x1="60" y1="292" x2="716" y2="292" stroke="#cbd5e1" stroke-width="2"/>
        <line x1="60" y1="96" x2="60" y2="292" stroke="#cbd5e1" stroke-width="2"/>
        {''.join(parts)}
    </svg>
    """


def _iteration_cycle_svg() -> str:
    return """
    <svg viewBox="0 0 900 470" width="100%" height="360" role="img" aria-label="Iterative improvement loop">
        <defs>
            <marker id="cycle-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="8" markerHeight="8" orient="auto">
                <path d="M 0 0 L 10 5 L 0 10 z" fill="#0f172a"/>
            </marker>
        </defs>
        <rect x="6" y="6" width="888" height="458" rx="28" fill="#ffffff" opacity="0.9"/>
        <circle cx="450" cy="235" r="58" fill="#14532d"/>
        <text x="450" y="226" text-anchor="middle" fill="#ffffff" font-size="18" font-weight="800">Human</text>
        <text x="450" y="248" text-anchor="middle" fill="#ecfdf5" font-size="13">decision gate</text>

        <rect x="378" y="44" width="144" height="62" rx="18" fill="#1d4ed8"/>
        <text x="450" y="71" text-anchor="middle" fill="#ffffff" font-size="14" font-weight="800">Retrieve context</text>
        <text x="450" y="90" text-anchor="middle" fill="#dbeafe" font-size="12">metrics + KB</text>

        <rect x="664" y="166" width="164" height="62" rx="18" fill="#7c3aed"/>
        <text x="746" y="193" text-anchor="middle" fill="#ffffff" font-size="14" font-weight="800">Patch proposal</text>
        <text x="746" y="212" text-anchor="middle" fill="#f5f3ff" font-size="12">reasoning or fallback model</text>

        <rect x="602" y="336" width="178" height="62" rx="18" fill="#f97316"/>
        <text x="691" y="363" text-anchor="middle" fill="#ffffff" font-size="14" font-weight="800">Stage + compare</text>
        <text x="691" y="382" text-anchor="middle" fill="#fff7ed" font-size="12">before / after deltas</text>

        <rect x="118" y="336" width="166" height="62" rx="18" fill="#0f766e"/>
        <text x="201" y="363" text-anchor="middle" fill="#ffffff" font-size="14" font-weight="800">Accept or reject</text>
        <text x="201" y="382" text-anchor="middle" fill="#ecfeff" font-size="12">history and rollback</text>

        <rect x="70" y="154" width="188" height="62" rx="18" fill="#475569"/>
        <text x="164" y="181" text-anchor="middle" fill="#ffffff" font-size="14" font-weight="800">Working design update</text>
        <text x="164" y="200" text-anchor="middle" fill="#f8fafc" font-size="12">metrics become next baseline</text>

        <path d="M 450 106 C 540 106, 650 124, 690 166" fill="none" stroke="#0f172a" stroke-width="3.5" marker-end="url(#cycle-arrow)"/>
        <path d="M 746 228 C 760 268, 748 314, 712 336" fill="none" stroke="#0f172a" stroke-width="3.5" marker-end="url(#cycle-arrow)"/>
        <path d="M 602 370 C 540 398, 390 398, 284 368" fill="none" stroke="#0f172a" stroke-width="3.5" marker-end="url(#cycle-arrow)"/>
        <path d="M 166 336 C 116 292, 102 252, 120 216" fill="none" stroke="#0f172a" stroke-width="3.5" marker-end="url(#cycle-arrow)"/>
        <path d="M 258 185 C 316 164, 372 176, 412 204" fill="none" stroke="#0f172a" stroke-width="3.5" marker-end="url(#cycle-arrow)"/>
    </svg>
    """


def _framework_flow_svg() -> str:
    return """
    <svg viewBox="0 0 1380 900" width="100%" height="760" role="img" aria-label="Full framework flowchart">
        <defs>
            <marker id="fw-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
                <path d="M 0 0 L 10 5 L 0 10 z" fill="#0f172a"/>
            </marker>
            <filter id="fw-shadow" x="-20%" y="-20%" width="140%" height="140%">
                <feDropShadow dx="0" dy="10" stdDeviation="12" flood-opacity="0.12"/>
            </filter>
        </defs>

        <rect x="6" y="6" width="1368" height="888" rx="34" fill="#fbfdff"/>
        <rect x="32" y="34" width="1316" height="68" rx="24" fill="#102117"/>
        <text x="60" y="62" fill="#ecfdf5" font-size="22" font-weight="800">General framework: lightweight cipher design, evaluation, improvement, and export</text>
        <text x="60" y="85" fill="#bbf7d0" font-size="13">The chart follows the same 6-stage research flow as the main app, with the copilot and export paths branching from the current working design.</text>

        <rect x="60" y="150" width="220" height="90" rx="24" fill="#14532d" filter="url(#fw-shadow)"/>
        <text x="170" y="184" text-anchor="middle" fill="#ffffff" font-size="18" font-weight="800">1. Build the spec</text>
        <text x="170" y="208" text-anchor="middle" fill="#ecfdf5" font-size="12">architecture, rounds, components</text>

        <rect x="340" y="150" width="220" height="90" rx="24" fill="#1d4ed8" filter="url(#fw-shadow)"/>
        <text x="450" y="184" text-anchor="middle" fill="#ffffff" font-size="18" font-weight="800">2. Validate + local metrics</text>
        <text x="450" y="208" text-anchor="middle" fill="#dbeafe" font-size="12">avalanche scores and issue flags</text>

        <rect x="620" y="150" width="220" height="90" rx="24" fill="#7c3aed" filter="url(#fw-shadow)"/>
        <text x="730" y="184" text-anchor="middle" fill="#ffffff" font-size="18" font-weight="800">3. Deep evaluation</text>
        <text x="730" y="208" text-anchor="middle" fill="#f5f3ff" font-size="12">roundtrip, SAC, S-box, I/O checks</text>

        <rect x="900" y="150" width="220" height="90" rx="24" fill="#9333ea" filter="url(#fw-shadow)"/>
        <text x="1010" y="184" text-anchor="middle" fill="#ffffff" font-size="18" font-weight="800">Diagnostics</text>
        <text x="1010" y="208" text-anchor="middle" fill="#f5f3ff" font-size="12">structured weaknesses and focus areas</text>

        <rect x="1180" y="150" width="140" height="90" rx="24" fill="#0f766e" filter="url(#fw-shadow)"/>
        <text x="1250" y="184" text-anchor="middle" fill="#ffffff" font-size="18" font-weight="800">KB</text>
        <text x="1250" y="208" text-anchor="middle" fill="#ccfbf1" font-size="12">retrieval</text>

        <rect x="1090" y="328" width="230" height="98" rx="24" fill="#f97316" filter="url(#fw-shadow)"/>
        <text x="1205" y="362" text-anchor="middle" fill="#ffffff" font-size="18" font-weight="800">Reasoning layer</text>
        <text x="1205" y="384" text-anchor="middle" fill="#fff7ed" font-size="12">reasoning model, quality fallback, fast fallback</text>
        <text x="1205" y="404" text-anchor="middle" fill="#fff7ed" font-size="12">returns ImprovementPatch or copilot answer</text>

        <rect x="780" y="328" width="240" height="98" rx="24" fill="#ea580c" filter="url(#fw-shadow)"/>
        <text x="900" y="362" text-anchor="middle" fill="#ffffff" font-size="18" font-weight="800">4. Stage the patch</text>
        <text x="900" y="384" text-anchor="middle" fill="#fff7ed" font-size="12">apply patch, detect mismatches, evolve if needed</text>
        <text x="900" y="404" text-anchor="middle" fill="#fff7ed" font-size="12">compute before / after deltas</text>

        <polygon points="548,360 650,300 752,360 650,420" fill="#475569" filter="url(#fw-shadow)"/>
        <text x="650" y="352" text-anchor="middle" fill="#ffffff" font-size="16" font-weight="800">Human</text>
        <text x="650" y="374" text-anchor="middle" fill="#ffffff" font-size="16" font-weight="800">accept?</text>

        <rect x="360" y="328" width="140" height="98" rx="24" fill="#0f766e" filter="url(#fw-shadow)"/>
        <text x="430" y="362" text-anchor="middle" fill="#ffffff" font-size="18" font-weight="800">Reject</text>
        <text x="430" y="384" text-anchor="middle" fill="#ccfbf1" font-size="12">keep current design</text>
        <text x="430" y="404" text-anchor="middle" fill="#ccfbf1" font-size="12">loop back to patch generation</text>

        <rect x="160" y="328" width="160" height="98" rx="24" fill="#14532d" filter="url(#fw-shadow)"/>
        <text x="240" y="362" text-anchor="middle" fill="#ffffff" font-size="18" font-weight="800">Accept</text>
        <text x="240" y="384" text-anchor="middle" fill="#dcfce7" font-size="12">promote after_spec to working spec</text>
        <text x="240" y="404" text-anchor="middle" fill="#dcfce7" font-size="12">recompute baseline metrics</text>
        <text x="444" y="294" text-anchor="middle" fill="#0f172a" font-size="13" font-weight="700">Yes</text>
        <text x="520" y="438" text-anchor="middle" fill="#0f172a" font-size="13" font-weight="700">No</text>

        <rect x="74" y="542" width="300" height="108" rx="26" fill="#1d4ed8" filter="url(#fw-shadow)"/>
        <text x="224" y="578" text-anchor="middle" fill="#ffffff" font-size="18" font-weight="800">Iteration history and rollback</text>
        <text x="224" y="600" text-anchor="middle" fill="#dbeafe" font-size="12">accepted / rejected records, reasons, deltas, checkpoints</text>
        <text x="224" y="620" text-anchor="middle" fill="#dbeafe" font-size="12">becomes context for the next loop</text>

        <rect x="430" y="542" width="292" height="108" rx="26" fill="#0f766e" filter="url(#fw-shadow)"/>
        <text x="576" y="578" text-anchor="middle" fill="#ffffff" font-size="18" font-weight="800">5. Export thesis artifacts</text>
        <text x="576" y="600" text-anchor="middle" fill="#ccfbf1" font-size="12">Python module, reproducible run, JSON history, LaTeX tables</text>
        <text x="576" y="620" text-anchor="middle" fill="#ccfbf1" font-size="12">supports reporting and reproducibility</text>

        <rect x="780" y="542" width="540" height="108" rx="26" fill="#7c2d12" filter="url(#fw-shadow)"/>
        <text x="1050" y="578" text-anchor="middle" fill="#ffffff" font-size="18" font-weight="800">6. Design-review copilot</text>
        <text x="1050" y="600" text-anchor="middle" fill="#ffedd5" font-size="12">context = working spec + metrics + diagnostics + history + retrieved KB evidence</text>
        <text x="1050" y="620" text-anchor="middle" fill="#ffedd5" font-size="12">chat can answer questions or emit PATCH_PROPOSAL for another staged review</text>

        <rect x="430" y="730" width="520" height="100" rx="26" fill="#102117" filter="url(#fw-shadow)"/>
        <text x="690" y="766" text-anchor="middle" fill="#ffffff" font-size="20" font-weight="800">Research outcome</text>
        <text x="690" y="790" text-anchor="middle" fill="#bbf7d0" font-size="13">A traceable path from initial design to evaluated improvements and thesis-ready outputs</text>

        <line x1="280" y1="195" x2="340" y2="195" stroke="#0f172a" stroke-width="4" marker-end="url(#fw-arrow)"/>
        <line x1="560" y1="195" x2="620" y2="195" stroke="#0f172a" stroke-width="4" marker-end="url(#fw-arrow)"/>
        <line x1="840" y1="195" x2="900" y2="195" stroke="#0f172a" stroke-width="4" marker-end="url(#fw-arrow)"/>
        <line x1="1120" y1="195" x2="1180" y2="195" stroke="#0f172a" stroke-width="4" marker-end="url(#fw-arrow)"/>

        <path d="M 1010 240 C 1010 282, 970 312, 930 328" fill="none" stroke="#0f172a" stroke-width="4" marker-end="url(#fw-arrow)"/>
        <path d="M 1250 240 C 1250 288, 1240 304, 1216 328" fill="none" stroke="#0f172a" stroke-width="4" marker-end="url(#fw-arrow)"/>
        <line x1="1090" y1="377" x2="1020" y2="377" stroke="#0f172a" stroke-width="4" marker-end="url(#fw-arrow)"/>
        <line x1="780" y1="377" x2="752" y2="377" stroke="#0f172a" stroke-width="4" marker-end="url(#fw-arrow)"/>
        <path d="M 612 323 C 560 286, 418 286, 280 328" fill="none" stroke="#0f172a" stroke-width="4" marker-end="url(#fw-arrow)"/>
        <line x1="548" y1="411" x2="500" y2="411" stroke="#0f172a" stroke-width="4" marker-end="url(#fw-arrow)"/>

        <path d="M 240 426 C 240 474, 240 500, 224 542" fill="none" stroke="#0f172a" stroke-width="4" marker-end="url(#fw-arrow)"/>
        <path d="M 430 426 C 430 474, 330 504, 280 536" fill="none" stroke="#0f172a" stroke-width="4" stroke-dasharray="9 8" marker-end="url(#fw-arrow)"/>
        <path d="M 900 426 C 900 474, 980 504, 1060 542" fill="none" stroke="#0f172a" stroke-width="4" marker-end="url(#fw-arrow)"/>
        <line x1="374" y1="596" x2="430" y2="596" stroke="#0f172a" stroke-width="4" marker-end="url(#fw-arrow)"/>
        <path d="M 224 650 C 224 694, 332 716, 430 748" fill="none" stroke="#0f172a" stroke-width="4" marker-end="url(#fw-arrow)"/>
        <path d="M 576 650 C 576 700, 610 718, 642 730" fill="none" stroke="#0f172a" stroke-width="4" marker-end="url(#fw-arrow)"/>
        <path d="M 1050 650 C 1050 700, 926 718, 850 730" fill="none" stroke="#0f172a" stroke-width="4" marker-end="url(#fw-arrow)"/>

        <path d="M 374 584 C 500 548, 636 500, 780 410" fill="none" stroke="#0f172a" stroke-width="4" stroke-dasharray="9 8" marker-end="url(#fw-arrow)"/>
        <path d="M 960 542 C 960 490, 1014 454, 1140 424" fill="none" stroke="#0f172a" stroke-width="4" stroke-dasharray="9 8" marker-end="url(#fw-arrow)"/>
    </svg>
    """


def _overview_html() -> str:
    step_pills = "".join(
        _pill(item)
        for item in [
            "3 cipher families",
            "6 research stages",
            "Deterministic evaluation",
            "Human-in-the-loop patch review",
            "RAG-assisted reasoning",
            "Thesis-ready exports",
        ]
    )
    return f"""
    <div class="app-shell">
        <div class="hero-card">
            <div class="hero-grid">
                <div>
                    <div class="eyebrow">Framework explainer</div>
                    <h1 class="hero-title">Lightweight cipher research workflow, presented as a guided visual map</h1>
                    <p class="hero-subtitle">
                        This Gradio UI is a compact companion to the main research app. It explains how the system
                        moves from cipher configuration to evaluation, controlled improvement, conversational review,
                        and reproducible thesis artifacts.
                    </p>
                    <div class="pill-row">{step_pills}</div>
                    <div class="kpi-grid">
                        <div class="kpi">
                            <div class="kpi-value">6</div>
                            <div class="kpi-label">Main UI stages mirrored from the research app</div>
                        </div>
                        <div class="kpi">
                            <div class="kpi-value">3</div>
                            <div class="kpi-label">Cipher families supported: SPN, FEISTEL, ARX</div>
                        </div>
                        <div class="kpi">
                            <div class="kpi-value">4+</div>
                            <div class="kpi-label">Output groups: module, run snapshot, JSON history, LaTeX tables</div>
                        </div>
                    </div>
                </div>
                <div class="diagram-card">
                    {_overview_map_svg()}
                </div>
            </div>
        </div>

        <div class="overview-grid" style="margin-top:18px;">
            <div class="section-card">
                <h3 class="section-title">What the app does</h3>
                <p class="section-copy">
                    It acts as a research workbench for lightweight block-cipher experiments. You configure a design,
                    evaluate it locally, diagnose weaknesses, ask for candidate improvements, review them, and keep a
                    traceable record of every decision.
                </p>
            </div>
            <div class="section-card">
                <h3 class="section-title">Where models fit</h3>
                <p class="section-copy">
                    Models assist only after the deterministic core has produced evidence. The reasoning model is used
                    when deeper design discussion or patch generation is needed. Fallback paths use quality or fast
                    models, and a code model is reserved for adaptive component repair.
                </p>
            </div>
            <div class="section-card">
                <h3 class="section-title">Why this layout matters</h3>
                <p class="section-copy">
                    The framework separates measurable evaluation from AI-assisted suggestions. That makes the thesis
                    story easier to defend: design choices are proposed by a model, but checked and accepted by a
                    deterministic pipeline plus a human decision gate.
                </p>
            </div>
        </div>
    </div>
    """


def _evaluation_html() -> str:
    return f"""
    <div class="app-shell">
        <div class="two-col-grid">
            <div class="section-card">
                <div class="eyebrow">Evaluation surface</div>
                <h2 class="section-title" style="margin-top:12px;">How the framework measures a design</h2>
                <p class="section-copy">
                    Section 2 is a cheap baseline pass, while section 3 adds correctness, SAC behavior,
                    S-box quality, and interface diagnostics. The outputs of both stages become evidence for
                    the iterative improvement loop and the design-review copilot.
                </p>
                <table class="mini-table">
                    <tr>
                        <th>Stage</th>
                        <td>Section 2: plaintext avalanche, key avalanche, heuristic issue flags</td>
                    </tr>
                    <tr>
                        <th>Deep checks</th>
                        <td>Section 3: roundtrip, SAC for plaintext and key, S-box DDT/LAT, I/O compatibility</td>
                    </tr>
                    <tr>
                        <th>Decision use</th>
                        <td>Before/after comparison, diagnostic prompts, accept or reject rationale, thesis tables</td>
                    </tr>
                </table>
                <div class="legend-row">
                    <span class="legend-chip"><span class="legend-dot" style="background:#1d4ed8;"></span>Local metrics</span>
                    <span class="legend-chip"><span class="legend-dot" style="background:#7c3aed;"></span>Advanced diagnostics</span>
                    <span class="legend-chip"><span class="legend-dot" style="background:#0f766e;"></span>History-aware comparison</span>
                </div>
            </div>
            <div class="diagram-card">
                {_signals_svg()}
            </div>
        </div>
        <div class="section-card" style="margin-top:18px;">
            <h3 class="section-title">Compact signal matrix</h3>
            <table class="mini-table">
                <tr>
                    <th>Signal</th>
                    <td>What it tells the researcher</td>
                    <td>Where it appears</td>
                </tr>
                <tr>
                    <th>Plaintext avalanche</th>
                    <td>Whether small plaintext changes spread through the ciphertext</td>
                    <td>Local metrics, staged patch comparison, history tables</td>
                </tr>
                <tr>
                    <th>Key avalanche</th>
                    <td>Whether small key changes meaningfully affect the ciphertext</td>
                    <td>Local metrics, staged patch comparison, history tables</td>
                </tr>
                <tr>
                    <th>Roundtrip</th>
                    <td>Correctness check for encryption and decryption symmetry</td>
                    <td>Advanced evaluation, diagnostics, feedback synthesis</td>
                </tr>
                <tr>
                    <th>SAC</th>
                    <td>Per-bit diffusion behavior around the ideal 0.50 response</td>
                    <td>Advanced evaluation, diagnostics, improvement analysis</td>
                </tr>
                <tr>
                    <th>S-box DDT/LAT</th>
                    <td>Differential and linear quality of substitution components</td>
                    <td>Advanced evaluation, diagnostics, feedback prompts</td>
                </tr>
                <tr>
                    <th>I/O compatibility</th>
                    <td>Whether component interfaces fit the chosen architecture and block width</td>
                    <td>Advanced evaluation, staged patch safety checks, adaptive evolution</td>
                </tr>
            </table>
        </div>
    </div>
    """


def _advanced_eval_flow_svg() -> str:
    return """
    <svg viewBox="0 0 1420 1150" width="100%" height="980" role="img" aria-label="Advanced evaluation execution flow">
        <defs>
            <marker id="ae-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
                <path d="M 0 0 L 10 5 L 0 10 z" fill="#0f172a"/>
            </marker>
            <filter id="ae-shadow" x="-20%" y="-20%" width="140%" height="140%">
                <feDropShadow dx="0" dy="10" stdDeviation="12" flood-opacity="0.12"/>
            </filter>
        </defs>

        <rect x="8" y="8" width="1404" height="1134" rx="34" fill="#fbfdff"/>
        <rect x="34" y="32" width="1352" height="74" rx="24" fill="#102117"/>
        <text x="62" y="61" fill="#ecfdf5" font-size="22" font-weight="800">Advanced evaluation: exact stage-3 execution path from the Streamlit app</text>
        <text x="62" y="86" fill="#bbf7d0" font-size="13">This view follows the current code path: setup, deterministic checks, report persistence, diagnostic extraction, optional feedback, and the separate I/O button lane.</text>

        <rect x="48" y="130" width="1324" height="160" rx="28" fill="#eff6ff" stroke="rgba(29, 78, 216, 0.10)"/>
        <text x="74" y="158" fill="#1d4ed8" font-size="15" font-weight="800">Trigger and setup</text>

        <rect x="74" y="182" width="220" height="86" rx="22" fill="#14532d" filter="url(#ae-shadow)"/>
        <text x="184" y="214" text-anchor="middle" fill="#ffffff" font-size="17" font-weight="800">Current working design</text>
        <text x="184" y="236" text-anchor="middle" fill="#dcfce7" font-size="12">valid CipherSpec already in session</text>

        <rect x="334" y="182" width="220" height="86" rx="22" fill="#1d4ed8" filter="url(#ae-shadow)"/>
        <text x="444" y="214" text-anchor="middle" fill="#ffffff" font-size="17" font-weight="800">Evaluation settings</text>
        <text x="444" y="236" text-anchor="middle" fill="#dbeafe" font-size="12">roundtrip vectors and SAC trials per bit</text>

        <rect x="594" y="182" width="220" height="86" rx="22" fill="#2563eb" filter="url(#ae-shadow)"/>
        <text x="704" y="214" text-anchor="middle" fill="#ffffff" font-size="17" font-weight="800">Run full evaluation</text>
        <text x="704" y="236" text-anchor="middle" fill="#dbeafe" font-size="12">section-3 button trigger</text>

        <rect x="854" y="182" width="220" height="86" rx="22" fill="#475569" filter="url(#ae-shadow)"/>
        <text x="964" y="214" text-anchor="middle" fill="#ffffff" font-size="17" font-weight="800">Create EvaluationReport</text>
        <text x="964" y="236" text-anchor="middle" fill="#e2e8f0" font-size="12">fresh report container for this run</text>

        <line x1="294" y1="225" x2="334" y2="225" stroke="#0f172a" stroke-width="4" marker-end="url(#ae-arrow)"/>
        <line x1="554" y1="225" x2="594" y2="225" stroke="#0f172a" stroke-width="4" marker-end="url(#ae-arrow)"/>
        <line x1="814" y1="225" x2="854" y2="225" stroke="#0f172a" stroke-width="4" marker-end="url(#ae-arrow)"/>

        <rect x="48" y="316" width="1324" height="238" rx="28" fill="#faf5ff" stroke="rgba(124, 58, 237, 0.10)"/>
        <text x="74" y="344" fill="#7c3aed" font-size="15" font-weight="800">Main deterministic evaluation sequence</text>

        <rect x="74" y="390" width="220" height="94" rx="22" fill="#1d4ed8" filter="url(#ae-shadow)"/>
        <text x="184" y="422" text-anchor="middle" fill="#ffffff" font-size="17" font-weight="800">Roundtrip verification</text>
        <text x="184" y="444" text-anchor="middle" fill="#dbeafe" font-size="12">random vectors, P = D(E(P,K),K)</text>

        <rect x="334" y="390" width="220" height="94" rx="22" fill="#2563eb" filter="url(#ae-shadow)"/>
        <text x="444" y="422" text-anchor="middle" fill="#ffffff" font-size="17" font-weight="800">Build cipher object</text>
        <text x="444" y="444" text-anchor="middle" fill="#dbeafe" font-size="12">instantiate the current working design</text>

        <rect x="594" y="390" width="180" height="94" rx="22" fill="#7c3aed" filter="url(#ae-shadow)"/>
        <text x="684" y="422" text-anchor="middle" fill="#ffffff" font-size="17" font-weight="800">SAC: plaintext</text>
        <text x="684" y="444" text-anchor="middle" fill="#f5f3ff" font-size="12">flip one plaintext bit per trial</text>

        <rect x="814" y="390" width="180" height="94" rx="22" fill="#8b5cf6" filter="url(#ae-shadow)"/>
        <text x="904" y="422" text-anchor="middle" fill="#ffffff" font-size="17" font-weight="800">SAC: key</text>
        <text x="904" y="444" text-anchor="middle" fill="#f5f3ff" font-size="12">flip one key bit per trial</text>

        <rect x="1034" y="390" width="272" height="94" rx="22" fill="#9333ea" filter="url(#ae-shadow)"/>
        <text x="1170" y="420" text-anchor="middle" fill="#ffffff" font-size="17" font-weight="800">Registry S-box profiling</text>
        <text x="1170" y="442" text-anchor="middle" fill="#f5f3ff" font-size="12">DDT, LAT, bijectivity on analyzable S-boxes</text>

        <path d="M 964 268 C 964 326, 364 350, 184 390" fill="none" stroke="#0f172a" stroke-width="4" marker-end="url(#ae-arrow)"/>
        <line x1="294" y1="437" x2="334" y2="437" stroke="#0f172a" stroke-width="4" marker-end="url(#ae-arrow)"/>
        <line x1="554" y1="437" x2="594" y2="437" stroke="#0f172a" stroke-width="4" marker-end="url(#ae-arrow)"/>
        <line x1="774" y1="437" x2="814" y2="437" stroke="#0f172a" stroke-width="4" marker-end="url(#ae-arrow)"/>
        <line x1="994" y1="437" x2="1034" y2="437" stroke="#0f172a" stroke-width="4" marker-end="url(#ae-arrow)"/>

        <rect x="48" y="582" width="1324" height="284" rx="28" fill="#fff7ed" stroke="rgba(249, 115, 22, 0.12)"/>
        <text x="74" y="610" fill="#ea580c" font-size="15" font-weight="800">Report, diagnostics, and optional feedback</text>

        <rect x="118" y="644" width="244" height="96" rx="22" fill="#0f766e" filter="url(#ae-shadow)"/>
        <text x="240" y="676" text-anchor="middle" fill="#ffffff" font-size="17" font-weight="800">Save `eval_report`</text>
        <text x="240" y="698" text-anchor="middle" fill="#ccfbf1" font-size="12">persist report in session state</text>

        <rect x="412" y="644" width="244" height="96" rx="22" fill="#0d9488" filter="url(#ae-shadow)"/>
        <text x="534" y="676" text-anchor="middle" fill="#ffffff" font-size="17" font-weight="800">Render summaries</text>
        <text x="534" y="698" text-anchor="middle" fill="#ccfbf1" font-size="12">roundtrip, SAC, and S-box results</text>

        <rect x="706" y="644" width="244" height="96" rx="22" fill="#f97316" filter="url(#ae-shadow)"/>
        <text x="828" y="676" text-anchor="middle" fill="#ffffff" font-size="17" font-weight="800">Parse diagnostics</text>
        <text x="828" y="698" text-anchor="middle" fill="#fff7ed" font-size="12">critical and warning findings</text>

        <rect x="998" y="612" width="298" height="72" rx="20" fill="#fb923c" filter="url(#ae-shadow)"/>
        <text x="1147" y="642" text-anchor="middle" fill="#ffffff" font-size="17" font-weight="800">Optional: Generate AI feedback</text>
        <text x="1147" y="662" text-anchor="middle" fill="#fff7ed" font-size="12">only if the researcher asks and an API is available</text>

        <rect x="998" y="706" width="138" height="66" rx="20" fill="#fdba74" filter="url(#ae-shadow)"/>
        <text x="1067" y="732" text-anchor="middle" fill="#7c2d12" font-size="15" font-weight="800">Retrieve KB</text>
        <text x="1067" y="751" text-anchor="middle" fill="#7c2d12" font-size="12">context</text>

        <rect x="1158" y="706" width="138" height="66" rx="20" fill="#f59e0b" filter="url(#ae-shadow)"/>
        <text x="1227" y="732" text-anchor="middle" fill="#ffffff" font-size="15" font-weight="800">Feedback call</text>
        <text x="1227" y="751" text-anchor="middle" fill="#fffbeb" font-size="12">reasoning or fallback model</text>

        <rect x="1040" y="786" width="214" height="70" rx="20" fill="#c2410c" filter="url(#ae-shadow)"/>
        <text x="1147" y="813" text-anchor="middle" fill="#ffffff" font-size="15" font-weight="800">Store `feedback_result`</text>
        <text x="1147" y="832" text-anchor="middle" fill="#ffedd5" font-size="12">patch, model used, reasoning trace</text>

        <path d="M 1170 484 C 1170 556, 344 592, 240 644" fill="none" stroke="#0f172a" stroke-width="4" marker-end="url(#ae-arrow)"/>
        <line x1="362" y1="692" x2="412" y2="692" stroke="#0f172a" stroke-width="4" marker-end="url(#ae-arrow)"/>
        <line x1="656" y1="692" x2="706" y2="692" stroke="#0f172a" stroke-width="4" marker-end="url(#ae-arrow)"/>
        <path d="M 950 676 C 970 664, 980 652, 998 648" fill="none" stroke="#0f172a" stroke-width="4" stroke-dasharray="10 8" marker-end="url(#ae-arrow)"/>
        <text x="972" y="632" fill="#9a3412" font-size="12" font-weight="700">if requested</text>
        <line x1="1147" y1="684" x2="1147" y2="706" stroke="#0f172a" stroke-width="4" marker-end="url(#ae-arrow)"/>
        <line x1="1136" y1="739" x2="1158" y2="739" stroke="#0f172a" stroke-width="4" marker-end="url(#ae-arrow)"/>
        <path d="M 1227 772 C 1227 790, 1202 800, 1147 786" fill="none" stroke="#0f172a" stroke-width="4" marker-end="url(#ae-arrow)"/>

        <rect x="48" y="892" width="1324" height="228" rx="28" fill="#f8fafc" stroke="rgba(71, 85, 105, 0.12)" stroke-dasharray="10 8"/>
        <text x="74" y="920" fill="#475569" font-size="15" font-weight="800">Separate button lane: I/O compatibility analysis</text>
        <text x="1094" y="920" fill="#64748b" font-size="12" font-weight="700">independent from the full-evaluation button</text>

        <rect x="94" y="968" width="248" height="86" rx="22" fill="#475569" filter="url(#ae-shadow)"/>
        <text x="218" y="1000" text-anchor="middle" fill="#ffffff" font-size="16" font-weight="800">Check I/O compatibility</text>
        <text x="218" y="1022" text-anchor="middle" fill="#e2e8f0" font-size="12">separate button in section 3</text>

        <rect x="394" y="968" width="272" height="86" rx="22" fill="#64748b" filter="url(#ae-shadow)"/>
        <text x="530" y="1000" text-anchor="middle" fill="#ffffff" font-size="16" font-weight="800">`detect_mismatches(...)`</text>
        <text x="530" y="1022" text-anchor="middle" fill="#e2e8f0" font-size="12">working_spec plus registry interface checks</text>

        <rect x="718" y="968" width="248" height="86" rx="22" fill="#1d4ed8" filter="url(#ae-shadow)"/>
        <text x="842" y="1000" text-anchor="middle" fill="#ffffff" font-size="16" font-weight="800">Show summaries</text>
        <text x="842" y="1022" text-anchor="middle" fill="#dbeafe" font-size="12">blocking errors, warnings, or pass</text>

        <rect x="1018" y="968" width="248" height="86" rx="22" fill="#0f766e" filter="url(#ae-shadow)"/>
        <text x="1142" y="1000" text-anchor="middle" fill="#ffffff" font-size="16" font-weight="800">Store `io_mismatches`</text>
        <text x="1142" y="1022" text-anchor="middle" fill="#ccfbf1" font-size="12">saved for later review and evolution</text>

        <path d="M 184 268 C 184 330, 144 378, 144 968" fill="none" stroke="#0f172a" stroke-width="4" stroke-dasharray="10 8" marker-end="url(#ae-arrow)"/>
        <line x1="342" y1="1011" x2="394" y2="1011" stroke="#0f172a" stroke-width="4" marker-end="url(#ae-arrow)"/>
        <line x1="666" y1="1011" x2="718" y2="1011" stroke="#0f172a" stroke-width="4" marker-end="url(#ae-arrow)"/>
        <line x1="966" y1="1011" x2="1018" y2="1011" stroke="#0f172a" stroke-width="4" marker-end="url(#ae-arrow)"/>
    </svg>
    """


def _advanced_evaluation_html() -> str:
    pills = "".join(
        _pill(item)
        for item in [
            "Current working spec",
            "Roundtrip correctness",
            "Plaintext SAC",
            "Key SAC",
            "Registry S-box profiling",
            "Diagnostics",
            "Optional feedback",
            "Separate I/O lane",
        ]
    )
    return f"""
    <div class="app-shell">
        <div class="section-card">
            <div class="eyebrow">Core methods</div>
            <h2 class="section-title" style="margin-top:12px;">The three main checks inside advanced evaluation</h2>
            <p class="section-copy">
                Advanced evaluation is not one metric. It combines correctness checking, deeper diffusion analysis,
                and component-level quality checks before the system turns the findings into diagnostics.
            </p>
            <div class="flow-grid" style="margin-top:16px;">
                <div class="flow-card">
                    <h4>1. Roundtrip verification</h4>
                    <ul>
                        <li><strong>Meaning:</strong> checks that decryption perfectly inverts encryption.</li>
                        <li><strong>In the system:</strong> many random plaintext/key pairs are generated and tested against the current working cipher.</li>
                        <li><strong>Why it matters:</strong> a failure means the design is functionally broken, so this is treated as the highest-priority issue.</li>
                    </ul>
                </div>
                <div class="flow-card">
                    <h4>2. SAC analysis</h4>
                    <ul>
                        <li><strong>Meaning:</strong> measures whether flipping one input bit changes about half of the output bits.</li>
                        <li><strong>In the system:</strong> SAC is run twice, once for plaintext bits and once for key bits, using repeated seeded trials.</li>
                        <li><strong>Why it matters:</strong> it shows whether diffusion is strong and whether weak bit positions or key-schedule problems exist.</li>
                    </ul>
                </div>
                <div class="flow-card">
                    <h4>3. Structural quality checks</h4>
                    <ul>
                        <li><strong>Meaning:</strong> examines substitution quality and interface fit.</li>
                        <li><strong>In the system:</strong> analyzable registry S-boxes are profiled with DDT, LAT, and bijectivity, while I/O compatibility is checked through a separate mismatch-analysis button path.</li>
                        <li><strong>Why it matters:</strong> it helps catch weak components and broken stage-to-stage wiring before improvement or export continues.</li>
                    </ul>
                </div>
            </div>
        </div>
        <div class="two-col-grid">
            <div class="section-card" style="margin-top:18px;">
                <div class="eyebrow">Stage 3 Zoom-In</div>
                <h2 class="section-title" style="margin-top:12px;">Advanced evaluation in one view</h2>
                <p class="section-copy">
                    This tab summarizes the advanced-evaluation stage without turning it into a long text explanation.
                    The diagram below follows the actual execution order in the Streamlit app: user settings, the
                    deterministic checks, report persistence, diagnostic parsing, optional feedback synthesis, and the
                    separate I/O compatibility button path.
                </p>
                <table class="mini-table">
                    <tr><th>Trigger</th><td>The researcher sets roundtrip and SAC counts, then clicks <code>Run full evaluation</code>.</td></tr>
                    <tr><th>Main sequence</th><td>Roundtrip -&gt; build cipher -&gt; SAC plaintext -&gt; SAC key -&gt; registry S-box profiling.</td></tr>
                    <tr><th>Saved state</th><td><code>eval_report</code>, parsed diagnostics, optional <code>feedback_result</code>, and separate <code>io_mismatches</code>.</td></tr>
                    <tr><th>Optional paths</th><td>AI feedback is only generated on request; I/O compatibility runs from its own button.</td></tr>
                </table>
                <div class="pill-row">{pills}</div>
            </div>
            <div class="detail-card" style="margin-top:18px;">
                <div class="flow-grid">
                    <div class="flow-card">
                        <h4>What enters stage 3</h4>
                        <ul>
                            <li>Current working design</li>
                            <li>User-selected roundtrip vector count</li>
                            <li>User-selected SAC trial count per bit</li>
                        </ul>
                    </div>
                    <div class="flow-card">
                        <h4>What the system does</h4>
                        <ul>
                            <li>Runs deterministic correctness and diffusion checks</li>
                            <li>Builds a structured evaluation report</li>
                            <li>Turns results into diagnostics for the next decision step</li>
                        </ul>
                    </div>
                    <div class="flow-card">
                        <h4>What comes out</h4>
                        <ul>
                            <li>Visible summaries in the UI</li>
                            <li>Stored report and diagnostics</li>
                            <li>Optional AI patch suggestion and separate I/O findings</li>
                        </ul>
                    </div>
                </div>
                <div class="tab-note">
                    The S-box block in the flowchart follows the current code path exactly: it profiles analyzable
                    registry S-boxes rather than only the active working-spec S-box.
                </div>
            </div>
        </div>
        <div class="diagram-card" style="margin-top:18px; padding:18px;">
            {_advanced_eval_flow_svg()}
        </div>
        <div class="legend-row">
            <span class="legend-chip"><span class="legend-dot" style="background:#1d4ed8;"></span>Inputs and button trigger</span>
            <span class="legend-chip"><span class="legend-dot" style="background:#7c3aed;"></span>Main deterministic evaluation lane</span>
            <span class="legend-chip"><span class="legend-dot" style="background:#0f766e;"></span>Saved report and diagnostics</span>
            <span class="legend-chip"><span class="legend-dot" style="background:#f97316;"></span>Optional feedback branch</span>
            <span class="legend-chip"><span class="legend-dot" style="background:#475569;"></span>Separate I/O compatibility branch</span>
        </div>
        <div class="tab-note">
            Solid arrows mark the main execution sequence. Dashed arrows mark optional or separately triggered paths.
        </div>
    </div>
    """


def _iteration_html() -> str:
    return f"""
    <div class="app-shell">
        <div class="two-col-grid">
            <div class="diagram-card">
                {_iteration_cycle_svg()}
            </div>
            <div class="section-card">
                <div class="eyebrow">Closed loop</div>
                <h2 class="section-title" style="margin-top:12px;">Why the improvement loop is more than chat</h2>
                <p class="section-copy">
                    The loop is structured so that each proposed change becomes a measurable experiment. A model
                    proposes a small patch, the system stages it, deterministic metrics compare before and after,
                    and the researcher still makes the final decision.
                </p>
                <table class="mini-table">
                    <tr><th>Patch source</th><td>Reasoning model or fallback model, grounded by metrics and KB context</td></tr>
                    <tr><th>Safety gates</th><td>Validation, mismatch detection, optional adaptive component evolution</td></tr>
                    <tr><th>Decision gate</th><td>Human acceptance or rejection, with a recorded rationale for traceability</td></tr>
                    <tr><th>Persistence</th><td>Accepted and rejected records, deltas, timestamps, and rollback checkpoints</td></tr>
                </table>
            </div>
        </div>
        <div class="section-card" style="margin-top:18px;">
            <h3 class="section-title">Role split inside one iteration</h3>
            <table class="mini-table">
                <tr>
                    <th>Researcher</th>
                    <td>Chooses when to generate, inspects the patch, decides accept or reject, and records the reason.</td>
                </tr>
                <tr>
                    <th>Deterministic engine</th>
                    <td>Builds the cipher, validates the patched design, runs metrics, computes deltas, and preserves the history.</td>
                </tr>
                <tr>
                    <th>Model assistance</th>
                    <td>Suggests candidate changes, synthesizes feedback from diagnostics, and can help repair component mismatches.</td>
                </tr>
            </table>
        </div>
    </div>
    """


def _coder_model_svg() -> str:
    return """
    <svg viewBox="0 0 1120 390" width="100%" height="320" role="img" aria-label="DeepSeek Coder model runtime flow">
        <defs>
            <marker id="coder-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto">
                <path d="M 0 0 L 10 5 L 0 10 z" fill="#0f172a"/>
            </marker>
        </defs>
        <rect x="8" y="8" width="1104" height="374" rx="28" fill="#f8fbff"/>
        <text x="44" y="44" fill="#0f172a" font-size="20" font-weight="800">Runtime role inside the framework</text>
        <text x="44" y="70" fill="#475569" font-size="12">This view shows how the coder model behaves when attached to the framework and KB on a local workstation or cloud machine.</text>

        <rect x="32" y="146" width="152" height="88" rx="20" fill="#1d4ed8"/>
        <text x="108" y="179" text-anchor="middle" fill="#ffffff" font-size="16" font-weight="800">Prompt + KB</text>
        <text x="108" y="201" text-anchor="middle" fill="#dbeafe" font-size="12">framework state</text>

        <rect x="232" y="146" width="162" height="88" rx="20" fill="#0f766e"/>
        <text x="313" y="179" text-anchor="middle" fill="#ffffff" font-size="16" font-weight="800">Tokenizer</text>
        <text x="313" y="201" text-anchor="middle" fill="#ccfbf1" font-size="12">embeddings</text>

        <rect x="442" y="90" width="250" height="200" rx="26" fill="#14532d"/>
        <text x="567" y="125" text-anchor="middle" fill="#ffffff" font-size="18" font-weight="800">Qwen2.5-Coder 7B decoder</text>
        <rect x="472" y="146" width="190" height="44" rx="14" fill="#22c55e"/>
        <text x="567" y="174" text-anchor="middle" fill="#052e16" font-size="14" font-weight="800">MLA attention block</text>
        <rect x="472" y="206" width="190" height="52" rx="14" fill="#86efac"/>
        <text x="567" y="232" text-anchor="middle" fill="#052e16" font-size="14" font-weight="800">DeepSeekMoE FFN</text>
        <text x="567" y="251" text-anchor="middle" fill="#14532d" font-size="11">shared experts + routed experts</text>

        <rect x="740" y="146" width="154" height="88" rx="20" fill="#7c3aed"/>
        <text x="817" y="179" text-anchor="middle" fill="#ffffff" font-size="16" font-weight="800">Instruction tuning</text>
        <text x="817" y="201" text-anchor="middle" fill="#ede9fe" font-size="12">code-oriented output</text>

        <rect x="942" y="146" width="146" height="88" rx="20" fill="#f97316"/>
        <text x="1015" y="179" text-anchor="middle" fill="#ffffff" font-size="16" font-weight="800">Code result</text>
        <text x="1015" y="201" text-anchor="middle" fill="#ffedd5" font-size="12">patches or modules</text>

        <line x1="184" y1="190" x2="232" y2="190" stroke="#0f172a" stroke-width="3.5" marker-end="url(#coder-arrow)"/>
        <line x1="394" y1="190" x2="442" y2="190" stroke="#0f172a" stroke-width="3.5" marker-end="url(#coder-arrow)"/>
        <line x1="692" y1="190" x2="740" y2="190" stroke="#0f172a" stroke-width="3.5" marker-end="url(#coder-arrow)"/>
        <line x1="894" y1="190" x2="942" y2="190" stroke="#0f172a" stroke-width="3.5" marker-end="url(#coder-arrow)"/>
    </svg>
    """


def _reasoning_model_svg() -> str:
    return """
    <svg viewBox="0 0 1120 390" width="100%" height="320" role="img" aria-label="DeepSeek reasoning model runtime flow">
        <defs>
            <marker id="reason-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto">
                <path d="M 0 0 L 10 5 L 0 10 z" fill="#0f172a"/>
            </marker>
        </defs>
        <rect x="8" y="8" width="1104" height="374" rx="28" fill="#fffaf7"/>
        <text x="44" y="44" fill="#0f172a" font-size="20" font-weight="800">Runtime role inside the framework</text>
        <text x="44" y="70" fill="#475569" font-size="12">This view shows how the reasoning model behaves when attached to the framework and KB on a local workstation or cloud machine.</text>

        <rect x="32" y="146" width="152" height="88" rx="20" fill="#7c2d12"/>
        <text x="108" y="179" text-anchor="middle" fill="#ffffff" font-size="16" font-weight="800">Prompt + KB</text>
        <text x="108" y="201" text-anchor="middle" fill="#ffedd5" font-size="12">design context</text>

        <rect x="232" y="146" width="162" height="88" rx="20" fill="#0f766e"/>
        <text x="313" y="179" text-anchor="middle" fill="#ffffff" font-size="16" font-weight="800">Tokenizer</text>
        <text x="313" y="201" text-anchor="middle" fill="#ccfbf1" font-size="12">embeddings</text>

        <rect x="442" y="90" width="250" height="200" rx="26" fill="#7c3aed"/>
        <text x="567" y="125" text-anchor="middle" fill="#ffffff" font-size="18" font-weight="800">Qwen2.5-7B decoder backbone</text>
        <rect x="472" y="146" width="190" height="44" rx="14" fill="#a78bfa"/>
        <text x="567" y="174" text-anchor="middle" fill="#1e1b4b" font-size="14" font-weight="800">Grouped-query attention</text>
        <rect x="472" y="206" width="190" height="52" rx="14" fill="#ddd6fe"/>
        <text x="567" y="232" text-anchor="middle" fill="#1e1b4b" font-size="14" font-weight="800">Dense SwiGLU FFN</text>
        <text x="567" y="251" text-anchor="middle" fill="#4338ca" font-size="11">stacked decoder transformer blocks</text>

        <rect x="740" y="146" width="154" height="88" rx="20" fill="#ea580c"/>
        <text x="817" y="179" text-anchor="middle" fill="#ffffff" font-size="16" font-weight="800">R1 distillation</text>
        <text x="817" y="201" text-anchor="middle" fill="#ffedd5" font-size="12">reasoning behavior</text>

        <rect x="942" y="146" width="146" height="88" rx="20" fill="#1d4ed8"/>
        <text x="1015" y="179" text-anchor="middle" fill="#ffffff" font-size="16" font-weight="800">Review / patch</text>
        <text x="1015" y="201" text-anchor="middle" fill="#dbeafe" font-size="12">analysis output</text>

        <line x1="184" y1="190" x2="232" y2="190" stroke="#0f172a" stroke-width="3.5" marker-end="url(#reason-arrow)"/>
        <line x1="394" y1="190" x2="442" y2="190" stroke="#0f172a" stroke-width="3.5" marker-end="url(#reason-arrow)"/>
        <line x1="692" y1="190" x2="740" y2="190" stroke="#0f172a" stroke-width="3.5" marker-end="url(#reason-arrow)"/>
        <line x1="894" y1="190" x2="942" y2="190" stroke="#0f172a" stroke-width="3.5" marker-end="url(#reason-arrow)"/>
    </svg>
    """


def _coder_base_arch_svg() -> str:
    return """
    <svg viewBox="0 0 1120 430" width="100%" height="340" role="img" aria-label="DeepSeek Coder base architecture">
        <defs>
            <marker id="coder-base-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto">
                <path d="M 0 0 L 10 5 L 0 10 z" fill="#0f172a"/>
            </marker>
        </defs>
        <rect x="8" y="8" width="1104" height="414" rx="28" fill="#f6fbf7"/>
        <text x="40" y="42" fill="#0f172a" font-size="20" font-weight="800">Base architecture: Qwen2.5-Coder 7B decoder backbone</text>
        <text x="40" y="68" fill="#475569" font-size="12">Officially this family is a decoder-only transformer optimized for code generation, code reasoning, and code repair.</text>

        <rect x="34" y="160" width="138" height="92" rx="20" fill="#1d4ed8"/>
        <text x="103" y="194" text-anchor="middle" fill="#ffffff" font-size="16" font-weight="800">Tokens</text>
        <text x="103" y="216" text-anchor="middle" fill="#dbeafe" font-size="12">code / text input</text>

        <rect x="224" y="160" width="166" height="92" rx="20" fill="#0f766e"/>
        <text x="307" y="194" text-anchor="middle" fill="#ffffff" font-size="16" font-weight="800">Embedding + RoPE</text>
        <text x="307" y="216" text-anchor="middle" fill="#ccfbf1" font-size="12">decoder input state</text>

        <rect x="444" y="96" width="302" height="220" rx="26" fill="#14532d"/>
        <text x="595" y="128" text-anchor="middle" fill="#ffffff" font-size="18" font-weight="800">Repeated decoder layers</text>
        <rect x="474" y="152" width="242" height="44" rx="14" fill="#22c55e"/>
        <text x="595" y="180" text-anchor="middle" fill="#052e16" font-size="14" font-weight="800">RMSNorm + GQA attention</text>
        <rect x="474" y="214" width="242" height="56" rx="14" fill="#86efac"/>
        <text x="595" y="240" text-anchor="middle" fill="#052e16" font-size="14" font-weight="800">RMSNorm + SwiGLU FFN</text>
        <text x="595" y="260" text-anchor="middle" fill="#14532d" font-size="11">dense feed-forward stack for code-oriented decoding</text>
        <text x="595" y="292" text-anchor="middle" fill="#dcfce7" font-size="12">repeat across the transformer stack</text>

        <rect x="800" y="160" width="126" height="92" rx="20" fill="#7c3aed"/>
        <text x="863" y="194" text-anchor="middle" fill="#ffffff" font-size="16" font-weight="800">Final norm</text>
        <text x="863" y="216" text-anchor="middle" fill="#ede9fe" font-size="12">decoder output</text>

        <rect x="976" y="160" width="112" height="92" rx="20" fill="#f97316"/>
        <text x="1032" y="194" text-anchor="middle" fill="#ffffff" font-size="16" font-weight="800">LM head</text>
        <text x="1032" y="216" text-anchor="middle" fill="#ffedd5" font-size="12">next token</text>

        <rect x="504" y="338" width="182" height="52" rx="16" fill="#dcfce7" stroke="#14532d"/>
        <text x="595" y="360" text-anchor="middle" fill="#14532d" font-size="13" font-weight="800">Key idea</text>
        <text x="595" y="378" text-anchor="middle" fill="#14532d" font-size="11">compact code-specialized instruction model</text>

        <line x1="172" y1="206" x2="224" y2="206" stroke="#0f172a" stroke-width="3.5" marker-end="url(#coder-base-arrow)"/>
        <line x1="390" y1="206" x2="444" y2="206" stroke="#0f172a" stroke-width="3.5" marker-end="url(#coder-base-arrow)"/>
        <line x1="746" y1="206" x2="800" y2="206" stroke="#0f172a" stroke-width="3.5" marker-end="url(#coder-base-arrow)"/>
        <line x1="926" y1="206" x2="976" y2="206" stroke="#0f172a" stroke-width="3.5" marker-end="url(#coder-base-arrow)"/>
    </svg>
    """


def _reasoning_base_arch_svg() -> str:
    return """
    <svg viewBox="0 0 1120 430" width="100%" height="340" role="img" aria-label="DeepSeek reasoning base architecture">
        <defs>
            <marker id="reason-base-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto">
                <path d="M 0 0 L 10 5 L 0 10 z" fill="#0f172a"/>
            </marker>
        </defs>
        <rect x="8" y="8" width="1104" height="414" rx="28" fill="#fff9f7"/>
        <text x="40" y="42" fill="#0f172a" font-size="20" font-weight="800">Base architecture: Qwen2.5-7B dense decoder backbone</text>
        <text x="40" y="68" fill="#475569" font-size="12">The distilled reasoning model keeps the Qwen2.5 decoder stack and adds DeepSeek-R1 distilled reasoning behavior through post-training.</text>

        <rect x="34" y="160" width="138" height="92" rx="20" fill="#7c2d12"/>
        <text x="103" y="194" text-anchor="middle" fill="#ffffff" font-size="16" font-weight="800">Tokens</text>
        <text x="103" y="216" text-anchor="middle" fill="#ffedd5" font-size="12">chat / analysis input</text>

        <rect x="224" y="160" width="166" height="92" rx="20" fill="#0f766e"/>
        <text x="307" y="194" text-anchor="middle" fill="#ffffff" font-size="16" font-weight="800">Embedding + RoPE</text>
        <text x="307" y="216" text-anchor="middle" fill="#ccfbf1" font-size="12">decoder input state</text>

        <rect x="444" y="96" width="302" height="220" rx="26" fill="#7c3aed"/>
        <text x="595" y="128" text-anchor="middle" fill="#ffffff" font-size="18" font-weight="800">Repeated decoder layers</text>
        <rect x="474" y="152" width="242" height="44" rx="14" fill="#a78bfa"/>
        <text x="595" y="180" text-anchor="middle" fill="#1e1b4b" font-size="14" font-weight="800">RMSNorm + GQA attention</text>
        <rect x="474" y="214" width="242" height="56" rx="14" fill="#ddd6fe"/>
        <text x="595" y="240" text-anchor="middle" fill="#1e1b4b" font-size="14" font-weight="800">RMSNorm + SwiGLU FFN</text>
        <text x="595" y="260" text-anchor="middle" fill="#4338ca" font-size="11">dense feed-forward stack, no MoE routing</text>
        <text x="595" y="292" text-anchor="middle" fill="#ede9fe" font-size="12">repeat across the transformer stack</text>

        <rect x="800" y="160" width="126" height="92" rx="20" fill="#ea580c"/>
        <text x="863" y="194" text-anchor="middle" fill="#ffffff" font-size="16" font-weight="800">Final norm</text>
        <text x="863" y="216" text-anchor="middle" fill="#ffedd5" font-size="12">decoder output</text>

        <rect x="976" y="160" width="112" height="92" rx="20" fill="#1d4ed8"/>
        <text x="1032" y="194" text-anchor="middle" fill="#ffffff" font-size="16" font-weight="800">LM head</text>
        <text x="1032" y="216" text-anchor="middle" fill="#dbeafe" font-size="12">next token</text>

        <rect x="474" y="338" width="242" height="52" rx="16" fill="#ffedd5" stroke="#7c2d12"/>
        <text x="595" y="360" text-anchor="middle" fill="#7c2d12" font-size="13" font-weight="800">Distillation layer on top</text>
        <text x="595" y="378" text-anchor="middle" fill="#7c2d12" font-size="11">R1-style reasoning behavior learned during post-training</text>

        <line x1="172" y1="206" x2="224" y2="206" stroke="#0f172a" stroke-width="3.5" marker-end="url(#reason-base-arrow)"/>
        <line x1="390" y1="206" x2="444" y2="206" stroke="#0f172a" stroke-width="3.5" marker-end="url(#reason-base-arrow)"/>
        <line x1="746" y1="206" x2="800" y2="206" stroke="#0f172a" stroke-width="3.5" marker-end="url(#reason-base-arrow)"/>
        <line x1="926" y1="206" x2="976" y2="206" stroke="#0f172a" stroke-width="3.5" marker-end="url(#reason-base-arrow)"/>
    </svg>
    """


def _model_integration_svg() -> str:
    return """
    <svg viewBox="0 0 1180 390" width="100%" height="315" role="img" aria-label="Model integration with framework and knowledge base">
        <defs>
            <marker id="model-link-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto">
                <path d="M 0 0 L 10 5 L 0 10 z" fill="#0f172a"/>
            </marker>
        </defs>
        <rect x="8" y="8" width="1164" height="374" rx="28" fill="#fcfcff"/>
        <rect x="36" y="148" width="210" height="108" rx="22" fill="#14532d"/>
        <text x="141" y="182" text-anchor="middle" fill="#ffffff" font-size="18" font-weight="800">Framework state</text>
        <text x="141" y="206" text-anchor="middle" fill="#dcfce7" font-size="12">spec, metrics, diagnostics, history</text>
        <text x="141" y="226" text-anchor="middle" fill="#dcfce7" font-size="12">pending patch, copilot context</text>

        <rect x="300" y="148" width="170" height="108" rx="22" fill="#0f766e"/>
        <text x="385" y="182" text-anchor="middle" fill="#ffffff" font-size="18" font-weight="800">KB retrieval</text>
        <text x="385" y="206" text-anchor="middle" fill="#ccfbf1" font-size="12">local index or cloud VM index</text>
        <text x="385" y="226" text-anchor="middle" fill="#ccfbf1" font-size="12">chunks attached to prompts</text>

        <rect x="530" y="86" width="260" height="108" rx="22" fill="#1d4ed8"/>
        <text x="660" y="120" text-anchor="middle" fill="#ffffff" font-size="18" font-weight="800">DeepSeek-R1-Distill-Qwen-7B</text>
        <text x="660" y="144" text-anchor="middle" fill="#dbeafe" font-size="12">local runtime or cloud inference node</text>
        <text x="660" y="164" text-anchor="middle" fill="#dbeafe" font-size="12">design review, patch ideas, explanation</text>

        <rect x="530" y="220" width="260" height="108" rx="22" fill="#7c3aed"/>
        <text x="660" y="254" text-anchor="middle" fill="#ffffff" font-size="18" font-weight="800">Qwen2.5-Coder-7B-Instruct</text>
        <text x="660" y="278" text-anchor="middle" fill="#ede9fe" font-size="12">local runtime or cloud inference node</text>
        <text x="660" y="298" text-anchor="middle" fill="#ede9fe" font-size="12">code edits, component work, exports</text>

        <rect x="850" y="148" width="292" height="108" rx="22" fill="#f97316"/>
        <text x="996" y="182" text-anchor="middle" fill="#ffffff" font-size="18" font-weight="800">System actions</text>
        <text x="996" y="206" text-anchor="middle" fill="#ffedd5" font-size="12">iterative improvement, copilot answers,</text>
        <text x="996" y="226" text-anchor="middle" fill="#ffedd5" font-size="12">component evolution, module generation</text>

        <line x1="246" y1="202" x2="300" y2="202" stroke="#0f172a" stroke-width="3.5" marker-end="url(#model-link-arrow)"/>
        <line x1="470" y1="202" x2="530" y2="140" stroke="#0f172a" stroke-width="3.5" marker-end="url(#model-link-arrow)"/>
        <line x1="470" y1="202" x2="530" y2="274" stroke="#0f172a" stroke-width="3.5" marker-end="url(#model-link-arrow)"/>
        <line x1="790" y1="140" x2="850" y2="182" stroke="#0f172a" stroke-width="3.5" marker-end="url(#model-link-arrow)"/>
        <line x1="790" y1="274" x2="850" y2="222" stroke="#0f172a" stroke-width="3.5" marker-end="url(#model-link-arrow)"/>

        <text x="40" y="42" fill="#0f172a" font-size="20" font-weight="800">Local or cloud inference path inside the framework</text>
        <text x="40" y="70" fill="#475569" font-size="12">The two models are shown as attached workers inside your system stack, fed by framework state and knowledge-base retrieval rather than exposed as external API calls.</text>
    </svg>
    """


def _models_html() -> str:
    return f"""
    <div class="app-shell">
        <div class="section-card">
            <div class="eyebrow">Model layer</div>
            <h2 class="section-title" style="margin-top:12px;">The two DeepSeek models as local or cloud workers in the framework</h2>
            <p class="section-copy">
                This view presents the two models as deployment choices inside your own stack. They sit beside the framework
                and the knowledge base, either on the local machine or on a cloud VM, and feed the same design-review and
                improvement workflow.
            </p>
            <table class="mini-table">
                <tr><th>Coder model</th><td>Qwen2.5-Coder-7B-Instruct: compact coder used for code-facing tasks, component work, and implementation-oriented outputs.</td></tr>
                <tr><th>Reasoning model</th><td>DeepSeek-R1-Distill-Qwen-7B: compact dense reasoning model used for design review, explanation, and patch planning.</td></tr>
                <tr><th>System link</th><td>Both consume framework context and retrieved KB chunks, then return outputs into the same iterative pipeline.</td></tr>
            </table>
        </div>

        <div class="diagram-card" style="margin-top:18px; padding:18px;">
            {_model_integration_svg()}
        </div>

        <div class="two-col-grid" style="margin-top:18px;">
            <div class="diagram-stack">
                <div class="diagram-card" style="padding:18px; margin-top:8px;">
                    {_coder_model_svg()}
                </div>
                <div class="diagram-card" style="padding:18px; margin-top:8px;">
                    {_coder_base_arch_svg()}
                </div>
            </div>
            <div class="section-card">
                <div class="eyebrow">Coder role</div>
                <h3 class="section-title" style="margin-top:12px;">Qwen2.5-Coder-7B-Instruct</h3>
                <p class="section-copy">
                    In the app story, this is the coding-side worker. It can be placed on a local runtime or a cloud machine
                    and tied to the framework state plus KB retrieval when code-oriented output is needed.
                </p>
                <table class="mini-table">
                    <tr><th>Total params</th><td>7.61B</td></tr>
                    <tr><th>Active params</th><td>Dense model, so no MoE active-parameter routing split</td></tr>
                    <tr><th>Context length</th><td>32,768 tokens for practical GGUF deployment</td></tr>
                    <tr><th>Variant type</th><td>Instruction-tuned coder model</td></tr>
                    <tr><th>Backbone</th><td>Qwen2.5 7B base family, specialized for code tasks</td></tr>
                    <tr><th>Family</th><td>Qwen2.5-Coder dense decoder architecture</td></tr>
                    <tr><th>Core pattern</th><td>Tokenizer and embeddings, repeated decoder blocks, grouped-query attention, dense feed-forward layers, final LM head</td></tr>
                    <tr><th>Use in framework</th><td>Code generation, component-oriented tasks, implementation-facing responses</td></tr>
                </table>
            </div>
        </div>

        <div class="two-col-grid" style="margin-top:18px;">
            <div class="section-card">
                <div class="eyebrow">Reasoning role</div>
                <h3 class="section-title" style="margin-top:12px;">DeepSeek-R1-Distill-Qwen-7B</h3>
                <p class="section-copy">
                    In the app story, this is the reasoning-side worker. It takes the current design state and KB evidence,
                    then returns analysis, review comments, and candidate improvement directions inside the same loop.
                </p>
                <table class="mini-table">
                    <tr><th>Total params</th><td>7B in the Qwen2.5-7B base family</td></tr>
                    <tr><th>Active params</th><td>Dense model, so no MoE active-parameter routing split</td></tr>
                    <tr><th>Context length</th><td>32,768 tokens for practical local reasoning runs</td></tr>
                    <tr><th>Generation length</th><td>DeepSeek-R1 evaluation uses up to 32,768 generated tokens</td></tr>
                    <tr><th>Variant type</th><td>Distilled reasoning model built from DeepSeek-R1 outputs</td></tr>
                    <tr><th>Backbone</th><td>Qwen2.5-7B dense decoder backbone</td></tr>
                    <tr><th>Family</th><td>Qwen2.5-7B dense decoder backbone with DeepSeek-R1 distillation</td></tr>
                    <tr><th>Core pattern</th><td>Tokenizer and embeddings, grouped-query attention, dense SwiGLU feed-forward blocks, final LM head, R1-style distilled reasoning behavior</td></tr>
                    <tr><th>Use in framework</th><td>Design review, improvement planning, patch rationale, context-aware discussion</td></tr>
                </table>
            </div>
            <div class="diagram-stack">
                <div class="diagram-card" style="padding:18px; margin-top:8px;">
                    {_reasoning_model_svg()}
                </div>
                <div class="diagram-card" style="padding:18px; margin-top:8px;">
                    {_reasoning_base_arch_svg()}
                </div>
            </div>
        </div>
    </div>
    """


def _artifacts_html() -> str:
    cards = [
        ("Standalone Python module", "Exports the current working design as a runnable cipher module with a self-test path.", "#1d4ed8"),
        ("Reproducible run snapshot", "Writes the spec, metrics, module, and iteration history into a timestamped directory.", "#14532d"),
        ("Iteration history JSON", "Preserves accepted and rejected patches, reasons, timestamps, and score deltas.", "#7c3aed"),
        ("LaTeX thesis tables", "Packages iteration summaries, accepted deltas, and roll-up statistics for publication.", "#0f766e"),
        ("Chat-supported review context", "Keeps the design-review copilot grounded in the same working design and history.", "#f97316"),
        ("Rollback checkpoints", "Allows the researcher to step back to an earlier accepted design when needed.", "#475569"),
    ]
    card_html = []
    for title, desc, color in cards:
        card_html.append(
            f"""
            <div class="artifact-card">
                <div class="legend-chip" style="display:inline-flex; margin-bottom:10px;">
                    <span class="legend-dot" style="background:{color};"></span>{html.escape(title)}
                </div>
                <p>{html.escape(desc)}</p>
            </div>
            """
        )
    return f"""
    <div class="app-shell">
        <div class="section-card">
            <div class="eyebrow">Research outputs</div>
            <h2 class="section-title" style="margin-top:12px;">What the framework leaves behind for the thesis</h2>
            <p class="section-copy">
                The export path turns live experiments into reusable evidence. Instead of only showing the final
                design, the system also preserves the path taken to get there.
            </p>
        </div>
        <div class="artifact-grid" style="margin-top:18px;">
            {''.join(card_html)}
        </div>
        <div class="section-card" style="margin-top:18px;">
            <h3 class="section-title">Artifact-to-thesis mapping</h3>
            <table class="mini-table">
                <tr><th>Artifact</th><td>Main thesis role</td></tr>
                <tr><th>Module export</th><td>Implementation evidence and reproducible code appendix</td></tr>
                <tr><th>Saved run folder</th><td>Experiment trace with the exact state used for a result</td></tr>
                <tr><th>Iteration history</th><td>Explains why patches were accepted or rejected</td></tr>
                <tr><th>LaTeX tables</th><td>Direct inclusion in methodology and results chapters</td></tr>
            </table>
        </div>
    </div>
    """


def build_demo() -> gr.Blocks:
    initial_step = STEP_DETAILS[0]["label"]
    with gr.Blocks(
        title="Cipher Lab Framework Explorer",
        analytics_enabled=False,
    ) as demo:
        gr.HTML(f"<style>{CUSTOM_CSS}</style>")
        with gr.Column(elem_classes="app-shell"):
            gr.HTML(_overview_html())
            with gr.Tabs():
                with gr.Tab("Overview"):
                    gr.HTML(
                        """
                        <div class="section-card">
                            <div class="eyebrow">System view</div>
                            <h2 class="section-title" style="margin-top:12px;">From design setup to thesis artifacts</h2>
                            <p class="section-copy">
                                Use this explainer as a fast visual guide to the framework. The next tabs walk through the
                                six stages, the evaluation surface, the closed-loop improvement logic, and the full flowchart.
                            </p>
                        </div>
                        """
                    )
                    gr.HTML(_step_strip())
                    gr.HTML(
                        """
                        <div class="tab-note">
                            Tip: the next tab lets you inspect each stage one by one with inputs, system work, outputs, and a mini flow.
                        </div>
                        """
                    )

                with gr.Tab("Step explorer"):
                    gr.HTML(
                        """
                        <div class="section-card">
                            <div class="eyebrow">Stage-by-stage</div>
                            <h2 class="section-title" style="margin-top:12px;">Explore every stage in the app workflow</h2>
                            <p class="section-copy">
                                Select a stage to see what enters it, what the system does, what comes out, and where model
                                assistance is involved.
                            </p>
                        </div>
                        """
                    )
                    gr.HTML(_step_strip())
                    with gr.Row():
                        step_choice = gr.Radio(
                            choices=[step["label"] for step in STEP_DETAILS],
                            value=initial_step,
                            label="Select a stage",
                        )
                    step_detail = gr.HTML(_render_step_detail(initial_step))
                    step_choice.change(fn=_render_step_detail, inputs=step_choice, outputs=step_detail)

                with gr.Tab("Evaluation and signals"):
                    gr.HTML(_evaluation_html())

                with gr.Tab("Advanced evaluation"):
                    gr.HTML(_advanced_evaluation_html())

                with gr.Tab("Improvement loop"):
                    gr.HTML(_iteration_html())

                with gr.Tab("Framework flowchart"):
                    gr.HTML(
                        """
                        <div class="section-card">
                            <div class="eyebrow">Full pipeline</div>
                            <h2 class="section-title" style="margin-top:12px;">General framework flowchart</h2>
                            <p class="section-copy">
                                This view compresses the full logic of the app into a single visual: build, validate, measure,
                                diagnose, retrieve context, propose, stage, decide, export, and review in chat.
                            </p>
                        </div>
                        """
                    )
                    gr.HTML(f"<div class='diagram-card' style='padding:18px;'>{_framework_flow_svg()}</div>")
                    gr.HTML(
                        """
                        <div class="legend-row">
                            <span class="legend-chip"><span class="legend-dot" style="background:#14532d;"></span>Design and accepted state</span>
                            <span class="legend-chip"><span class="legend-dot" style="background:#1d4ed8;"></span>Deterministic baseline checks</span>
                            <span class="legend-chip"><span class="legend-dot" style="background:#7c3aed;"></span>Diagnostics and deep analysis</span>
                            <span class="legend-chip"><span class="legend-dot" style="background:#f97316;"></span>Model-assisted proposal path</span>
                            <span class="legend-chip"><span class="legend-dot" style="background:#0f766e;"></span>Export and evidence outputs</span>
                        </div>
                        """
                    )

                with gr.Tab("Models in the framework"):
                    gr.HTML(_models_html())

                with gr.Tab("Artifacts and thesis view"):
                    gr.HTML(_artifacts_html())

    return demo


demo = build_demo()


def launch_demo() -> None:
    launch_sig = inspect.signature(demo.launch)
    launch_kwargs = {}
    if "show_api" in launch_sig.parameters:
        launch_kwargs["show_api"] = False
    demo.launch(**launch_kwargs)


if __name__ == "__main__":
    launch_demo()
