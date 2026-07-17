# 🔬 SciFlow Agent

SciFlow Agent is a lightweight agentic scientific-image-analysis application. You upload or select a
2D scientific image, describe the analysis in plain language, review an automatically generated
processing plan, approve it, and get segmentation results, object measurements, and a downloadable
reproducibility report.

The language model never executes code. It may only select **approved tools with validated
parameters** from a fixed registry — every plan is schema-checked before anything runs.

> **Status:** In development — Phase 1 (project foundation) complete.
> The full MVP workflow (planning → validation → approval → execution → report) lands in
> subsequent phases.

## Planned MVP workflow

```text
Select or upload an image
        ↓
"Remove noise, segment the bright objects,
 ignore very small regions, and measure them."
        ↓
Planner (demo rules or LLM) → structured plan
        ↓
Validation against the approved tool registry
        ↓
You review and approve the plan
        ↓
Controlled execution → mask, overlay, measurements
        ↓
Downloadable reproducibility report
```

## Approved tools

`convert_to_grayscale` · `denoise_median` · `enhance_contrast` · `segment_otsu` · `clean_mask` ·
`measure_objects`

## Quickstart

Requires **Python 3.11+**.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

The app opens at `http://localhost:8501` and works out of the box in **demo mode** — no API key
required. Two built-in example images are included.

## Configuration

Copy `.env.example` to `.env` and adjust as needed. Key settings:

| Variable | Default | Purpose |
|---|---|---|
| `PLANNER_MODE` | `demo` | `demo` (deterministic, offline) or `llm` |
| `LLM_BASE_URL` | — | OpenAI-compatible endpoint (LLM mode only) |
| `LLM_API_KEY` | — | API key (never logged or included in reports) |
| `LLM_MODEL` | — | Model name |
| `MAX_WORKFLOW_STEPS` | `8` | Safety cap on plan length |
| `MAX_IMAGE_WIDTH` / `MAX_IMAGE_HEIGHT` | `4096` | Maximum accepted image size |

## Development

```bash
pip install -r requirements-dev.txt
pytest              # run tests
ruff check .        # lint
ruff format .       # format
```

The built-in example images are generated deterministically:

```bash
python examples/generate_examples.py
```

## Project structure

```text
app.py                  Streamlit entry point
src/
  agent/                Planners, prompts, plan schemas
  tools/                Approved image-processing tools
  config.py             Central configuration (env-based)
  executor.py           Controlled workflow executor (Phase 3)
  tool_registry.py      Fixed tool registry (Phase 3)
  reporting.py          Reproducibility reports (Phase 7)
  visualization.py      Overlays and labelled views (Phase 2)
examples/               Built-in example images + generator
tests/                  Unit and integration tests
benchmark/              Synthetic benchmark (Phase 8)
```

## Safety model

- Fixed tool registry — the planner can only pick from a whitelist.
- Pydantic validation of every plan, tool name, and parameter.
- Bounded parameters and a maximum workflow length.
- No `eval`, no `exec`, no shell, no dynamic imports, no filesystem paths from model output.
- Explicit human approval before any execution.
- Secrets live in environment variables and are excluded from logs and reports.
