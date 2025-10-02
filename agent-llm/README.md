# AgenTest – LLM Component (UI + Service + Backend) (MVP)

This folder contains the **LLM side** of AgenTest: a minimal, local-first stack that turns a natural-language **Test Step** into an **action plan** for a Windows SUT (System Under Test), executes it, and performs a simple verification.

**Components**
- `ui.py` — Streamlit UI (enter one step, run, view logs & result).
- `llm_service.py` — FastAPI HTTP service wrapping the backend (`/healthz`, `/plan_step`, `/run_step`, plus optional `/state`, `/action` proxies).
- `llm_backend.py` — Core orchestration: **SUT `/state` → LLM prompt → action plan → SUT `/action` → SUT `/state` → heuristic verify**.
- `plan_utils.py` — LLM'den gelen action planlarını normalize eden, doğrulayan ve SUT'a güvenli şekilde iletilmesini sağlayan güvenlik/guardrail katmanı. (Tüm action planları burada schema'ya uygun hale getirilir, hatalı adımlar filtrelenir, timing guardrails eklenir.)
- `requirements.txt` — Python dependencies.
- `.env` — Environment variables (OpenRouter key, model, SUT URLs, service base URL).

```bash
.
├─ ui.py              # Streamlit UI
├─ llm_service.py     # FastAPI app (HTTP API)
├─ llm_backend.py     # Orchestration (SUT + LLM + verify)
├─ plan_utils.py      # Action plan security/guardrail layer
├─ requirements.txt   # Dependencies
└─ .env               # Local config (see example)
``` 
---

## 1 Prerequisites

- **Python 3.10+** (tested with 3.12)
- **SUT** running locally (lab exe) on `http://127.0.0.1:18080` with:
  - `POST /state` → returns current UI state (elements + screen meta; optional screenshot base64)
  - `POST /action` → executes an action plan and returns ACK (`{"status":"ok","applied":N}`)
- **OpenRouter API key** (free model supported): https://openrouter.ai

> You can test without a real SUT using the **Mock SUT** below.

---

## 2 Install

```bash
# Create and activate a virtual environment
python -m venv .venv
# Windows PowerShell:
. .venv/Scripts/Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt  
``` 

## 3 Configure
Create a .env file in this folder 
```bash
# OpenRouter
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxx
OPENROUTER_MODEL=openai/gpt-oss-20b:free

# SUT endpoints (your lab agent)
SUT_STATE_URL=http://127.0.0.1:18080/state
SUT_ACTION_URL=http://127.0.0.1:18080/action

# UI -> Service
LLM_SERVICE_BASE=http://127.0.0.1:18081
``` 

## 4 Run 

**Start the service (FastAPI)**

``` bash
uvicorn llm_service:app --host 0.0.0.0 --port 18081 --reload
```

Docs: http://127.0.0.1:18081/docs

Health: GET /healthz

**Start the UI (Streamlit)**
``` bash
streamlit run ui.py
```

Open the URL printed by Streamlit, fill Step, Expected Result, (optional Note) and click Run Step.

## 5 How it works

UI → Service /run_step

Service → Backend:

* POST SUT /state

* Build strict system prompt + context (elements + geometry)

* Call OpenRouter model (default: openai/gpt-oss-20b:free) → JSON action plan

* POST SUT /action with the plan

* Short wait → POST SUT /state again

* Simple heuristic verify (keyword hits in element names)

Service returns logs, verification, and the action plan to the UI.

``` bash

[Client] ──> Streamlit UI (ui.py)
                  │   (HTTP)
                  ▼
           FastAPI Service (llm_service.py)
                  │   calls
                  ▼
          LLM Backend (llm_backend.py)
        ┌──────────┴──────────┐
        │                     │
   OpenRouter (LLM)       SUT (lab exe)
      /chat/completions     /state  /action
```