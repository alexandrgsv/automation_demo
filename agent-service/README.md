# Edge Agent Service

Small FastAPI service with three deterministic demos: an edge-config FPS check,
a PR diff audit, and a developer-versus-QA loop. No GPU and no API key are
required. An empty `LLM_API_KEY` keeps every path on fixed rules.

Open [http://localhost:8000/docs](http://localhost:8000/docs) after startup. The service root redirects there.

## Architecture

```text
Client
  -> FastAPI                         /health  /api/v1/run  /api/v1/stream  /api/v1/pr-review  /api/v1/adversarial-review
    -> Skill contract                schema.py in the skill directory
      -> Agent orchestrator          sliding window, one device repair, mock fallback
        -> Skill                     prompt.md + logic.py
             edge_fps_evaluator      evaluate_edge_config
             pr_code_reviewer        analyze_pr_diff, generate_pytest_stub, calculate_review_risk_score
             adversarial_review      execute_in_sandbox
```

A skill is the contract, the protocol, and the function. `schema.py` rejects a bad payload. `prompt.md` says when to call the skill and how a device alias is repaired. `logic.py` is the deterministic function. `app/tools/` is only the registry that dispatches those functions.

`POST /api/v1/adversarial-review` runs two fixed snippets. The first divides by
duration and fails the QA script with `ZeroDivisionError`. The second rejects a
non-positive duration and the same script passes. The request `task` is only a
label. Caller source is not executed.

`POST /api/v1/pr-review` always runs the three review tools in that order.
Zero issues is Low and `approve`. One issue is Medium. Two or more is High.
Medium and High both return `request_changes`. The saved-time figure is fixed at 45 minutes.

`POST /api/v1/run` returns HTTP 422 when the body is not a `RunRequest`.
A body whose `tool_arguments` fail `BenchmarkInput` still returns HTTP 200.
The failure is a `validation_error` inside `tool_invocations`, and the process stays up.
`device=GPU` is rebuilt once to `cuda`. `tpu`, a zero batch, or an unknown field is not invented away.

Session memory is an in-process sliding window (`MEMORY_WINDOW`, default 8 messages).
A restart clears it. Pass `messages` to replace that window for one call.

## Layout

```text
agent-service/
├── app/
│   ├── main.py                      # /health, /run, /stream, /pr-review, /adversarial-review
│   ├── config.py                    # Pydantic Settings from .env
│   ├── schemas.py                   # HTTP envelopes; re-exports skill contracts
│   ├── agent.py                     # dispatch loop, applies the skill repair once
│   ├── skills/
│   │   ├── base.py                  # name, contract, prompt, repair, execute
│   │   ├── edge_fps_evaluator/      # schema.py, prompt.md, logic.py
│   │   ├── pr_code_reviewer/        # schema.py, prompt.md, logic.py
│   │   └── adversarial_review/      # schema.py, prompt.md, logic.py
│   └── tools/                       # registry over the skill functions
├── tests/
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

## Benchmark formula

This is a fixed stand-in, not a profiler. The same payload always prints the same FPS.
Swap the body of `evaluate_edge_config` for a real measurement without changing the schema.

`estimated_latency_ms = base_latency_ms[device] * batch_size`

`estimated_fps = 1000 / estimated_latency_ms`

Batching does not overlap work. A larger batch only stacks latency.

| device | base latency |
| --- | --- |
| cpu | 45 ms |
| cuda | 12.5 ms |
| tensorrt | 6 ms |

A pass is `estimated_fps >= target_fps`. Otherwise `bottleneck_detected` is true.

Demo numbers for batch 2 on cuda with a target of 30 FPS: latency 25 ms, **40 FPS**, pass.

## Quick start

```bash
cp .env.example .env
docker compose up --build
```

Leave `LLM_API_KEY` empty for the demo. Compose loads `.env` into the container.
The healthcheck calls `GET /health` from inside the container.

Without Docker, from this directory:

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python -m app.main
```

## Sample payload

```bash
curl -s http://localhost:8000/api/v1/run \
  -H 'Content-Type: application/json' \
  -d '{
    "message": "Check this edge config against the FPS target.",
    "session_id": "demo-1",
    "tool_arguments": {"batch_size": 2, "device": "cuda", "target_fps": 30}
  }'
```

```json
{
  "result": "Config device=cuda batch_size=2 estimates 40.0 FPS (latency 25.0 ms) and meets the target of 30.0 FPS. No bottleneck was detected.",
  "steps_taken": 2,
  "tool_invocations": [
    {
      "tool_name": "evaluate_edge_config",
      "arguments": {"batch_size": 2, "device": "cuda", "target_fps": 30},
      "ok": true,
      "output": {
        "device": "cuda",
        "batch_size": 2,
        "estimated_latency_ms": 25.0,
        "estimated_fps": 40.0,
        "target_fps": 30.0,
        "meets_requirements": true,
        "bottleneck_detected": false
      },
      "error": null
    }
  ],
  "latency_ms": 1.2,
  "mode": "mock",
  "session_id": "demo-1"
}
```

`latency_ms` is measured per call. The other fields are fixed for this payload.

A device the schema does not know stays a failed invocation and still returns HTTP 200:

```bash
curl -s http://localhost:8000/api/v1/run \
  -H 'Content-Type: application/json' \
  -d '{"message":"Unknown target.","tool_arguments":{"batch_size":1,"device":"tpu","target_fps":30}}'
```

`POST /api/v1/stream` emits the same turn as `text/event-stream` events named `step`, `tool`, and `result`.

## PR triage sample

```bash
curl -s http://localhost:8000/api/v1/pr-review \
  -H 'Content-Type: application/json' \
  -d '{
    "pr_id": 42,
    "file_path": "app/agent.py",
    "diff_content": "def run(data):\n    try:\n        return data\n    except:\n        pass\n# TODO: narrow the handler\n"
  }'
```

The audit reports High risk, `can_auto_merge: false`, and these test names:
`test_agent_edge_cases`, `test_agent_invalid_schema_handling`.
Checks, in order: a `def` with no `->` anywhere in the text, a bare `except:` or `except Exception`, then `TODO` or `FIXME`.

## Developer versus QA sample

```bash
curl -s http://localhost:8000/api/v1/adversarial-review \
  -H 'Content-Type: application/json' \
  -d '{"task":"Implement edge throughput calculation with zero-division safety"}'
```

`runs[0]` fails with `ZeroDivisionError`. `runs[1]` passes. `final_status` is `Passed`.

## What the trace is for

- One HTTP call replaces a manual pass over batch size, device, and target FPS.
- One HTTP call runs the diff checks, the pytest stub, and the merge score.
- Tool arguments are checked with `model_validate` before the function runs.
- A validation failure is data in the response, not an unhandled exception.
- Mock mode is the fallback when the key is missing or the provider call fails, so the demo does not depend on an external balance.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```
