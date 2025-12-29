# AgenTest - Agent-LLM

## Quick Start
```bash
# Backend
uvicorn llm_service:app --port 18888 --reload

# UI
cd ui && npm run dev

# Ollama (if using)
ollama serve
```

## LLM Providers

| Provider | Base URL | API Key | JSON Mode |
|----------|----------|---------|-----------|
| `ollama` | `http://localhost:11434` | No | `format: "json"` |
| `lmstudio` | `http://localhost:1234/v1` | No | `response_format` |
| `openrouter` | `https://openrouter.ai/api/v1` | Yes | `response_format` |
| `openai` | `https://api.openai.com/v1` | Yes | `response_format` |
| `anthropic` | - | Yes | Prompt-based |
| `gemini` | - | Yes | `response_mime_type` |
| `custom` | User-defined | Optional | `response_format` |

**JSON Enforcement:** All providers use native JSON mode where available + schema in SYSTEM_PROMPT.

**Priority:** UI Settings > `.env` > Defaults

## Settings Storage
```
~/.agentest/
├── llm_settings.json  # Encrypted settings
└── .key               # Fernet encryption key
```
Auto-created on first save. API keys are encrypted at rest.

## Execution Flow (Function Calls)

```
UI: runScenario() / replayTest()
    ↓
llm_service.py:
    POST /run → run_scenario_endpoint()
        ↓
    _mk_backend() → Creates LLMBackend with settings
        ↓
llm_backend.py:
    LLMBackend.run_scenario(steps)
        ↓
        for each step:
            ↓
    LLMBackend.run_step(test_step, expected_result)
        ↓
        ┌─────────────────────────────────────────┐
        │ ATTEMPT LOOP (max_attempts times)       │
        ├─────────────────────────────────────────┤
        │ 1. _fetch_state()                       │
        │    → GET state from WinDriver/ODS      │
        │    → SemanticFilter.filter_elements()   │
        │                                         │
        │ 2. _build_messages()                    │
        │    → Format state as LLM prompt         │
        │                                         │
        │ 3. _request_plan()                      │
        │    → _request_plan_ollama()             │
        │    → _request_plan_openai()             │
        │    → _request_plan_gemini()             │
        │    → _request_plan_anthropic()          │
        │    → Returns JSON action plan           │
        │                                         │
        │ 4. _parse_plan()                        │
        │    → Extract JSON from LLM response     │
        │                                         │
        │ 5. _send_action(plan)                   │
        │    → POST action to SUT controller     │
        │                                         │
        │ 6. _expected_holds() [VALIDATION]       │
        │    → _detect_ui_change(before, after)   │
        │    → ExpectedResultValidator.validate() │
        │      ├─ _parse_conditions()             │
        │      ├─ _match_targets() [fuzzy]        │
        │      └─ _validate_single_condition()    │
        │                                         │
        │ 7. If PASS → return success             │
        │    If FAIL → next attempt (ODS fallback)│
        └─────────────────────────────────────────┘
        ↓
    _save_merged_recording() → saved_tests/*.json
```

## Key Files
- `llm_backend.py` - LLM providers, validation, action execution
- `llm_service.py` - FastAPI endpoints, settings API
- `settings_manager.py` - Secure credential storage 
- `semantic_filter.py` - Element filtering for LLM
