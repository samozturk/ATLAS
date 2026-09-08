# ATLAS

ATLAS is a reliable, event-driven home AI system, built incrementally from a
local assistant into a proactive system for the home. This repository currently
contains the Phase 3 local-agent foundation: configuration, structured logging,
typed events, health checks, streaming Ollama chat, an application-owned tool
registry, and a bounded read-only tool loop.

## Development approach

The project follows the development roadmap in small, independently useful
phases. The current boundaries are intentional:

- `config.py` is the single place where environment-driven runtime configuration
  enters the application. Future integrations should receive typed settings
  rather than reading environment variables themselves.
- `events.py` defines a transport-neutral event envelope. MQTT, NATS, tools, and
  internal services can use this same contract without the core becoming tied to
  a particular broker.
- `main.py` owns application lifecycle and HTTP concerns only. Reasoning,
  device adapters, storage, and tool execution will live in their own modules as
  they are introduced.
- Liveness and readiness are separate: a running process is not necessarily
  ready to serve work. This gives future dependencies—such as a database or
  event bus—a safe place to participate in startup.

This separation supports the roadmap's reliability requirement: simple home
automation must remain deterministic and functional when the LLM or an
external service is unavailable. The LLM is behind a provider abstraction and
is used for interpretation only where rules cannot safely make the decision.
The tool registry stays application-owned: the model may request a tool, but
ATLAS validates its arguments, applies an execution timeout, executes only a
registered tool, and returns a controlled result to the model. The current
tools are read-only local time and opt-in weather; notes and home-device actions
remain out of scope until their data source and permission policy are explicit.

## Dependencies and Python version

ATLAS uses [uv](https://docs.astral.sh/uv/) for Python version, package, and
lockfile management. Do not create a virtual environment with `venv` or install
project dependencies with `pip`; use `uv` so every developer and deployment
resolves the same dependency set.

Install uv once if it is not already available:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

The project requires Python 3.12 or newer. The committed `.python-version`
pins local development to the 3.12 release line; `uv` will select an installed
compatible interpreter, or can install one when asked:

```bash
uv python install 3.12
```

## Run locally

```bash
cp .env.example .env
uv sync --all-extras
uv run atlas
```

ATLAS listens on `http://localhost:8000`. Check it with:

```bash
curl http://localhost:8000/health/live
curl http://localhost:8000/health/ready
```

## Local chat with Ollama

Install and start [Ollama](https://ollama.com), then pull both configured model
roles:

```bash
ollama pull qwen3.6:35b
ollama pull qwen3.5:122b-a10b
ollama serve
```

ATLAS uses the fast brain (`qwen3.6:35b`) by default. Select the deep brain
(`qwen3.5:122b-a10b`) only for complex requests; it has substantially higher
memory and startup requirements. The tags are configured with `ATLAS_FAST_MODEL`
and `ATLAS_DEEP_MODEL` in `.env`.

Start ATLAS, then send a non-streaming chat request:

```bash
curl http://localhost:8000/chat \
  -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello, ATLAS."}]}'
```

Choose the deep brain explicitly when needed:

```bash
curl http://localhost:8000/chat \
  -H 'content-type: application/json' \
  -d '{"brain":"deep","messages":[{"role":"user","content":"Compare two approaches in detail."}]}'
```

For server-sent streaming events, use `/chat/stream` instead:

```bash
curl -N http://localhost:8000/chat/stream \
  -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"Give me a short greeting."}]}'
```

Set `ATLAS_OLLAMA_BASE_URL` when Ollama is hosted on a different machine. Change
`ATLAS_FAST_MODEL` or `ATLAS_DEEP_MODEL` only to an installed Ollama tag.
Request/connect timeouts and retry settings are listed in `.env.example`.

## Tool calling

ATLAS supplies safe, read-only tools. `get_current_time` and
`get_current_weather` are always available. Weather defaults to Waalwijk,
Netherlands when no place is named. When the user asks about another place,
ATLAS resolves that name with [Open-Meteo's geocoding API](https://open-meteo.com/en/docs/geocoding-api),
then queries current conditions with the resolved coordinates. Results are
cached for ten minutes by default.

Set a different default home place in `.env` if desired:

```bash
ATLAS_WEATHER_DEFAULT_LOCATION=<home-city-and-country>
```

Weather is an external request: the default or user-requested place is sent to
the provider for geocoding and forecast retrieval. `ATLAS_WEATHER_REQUEST_TIMEOUT_SECONDS` and
`ATLAS_WEATHER_CACHE_TTL_SECONDS` control its bounded request and cache.

Each chat turn allows at most three model-to-tool rounds by default. Configure
`ATLAS_AGENT_MAX_TOOL_ROUNDS` and `ATLAS_TOOL_EXECUTION_TIMEOUT_SECONDS` in
`.env` only when a future tool requires different limits.

## Chat interface

The React interface lives in `frontend/`. It is a local companion to the ATLAS
API: its development server proxies chat requests to ATLAS, so the browser never
needs a separate cross-origin configuration. `docker compose up --build` starts
both services and exposes the interface at `http://localhost:3001`.

For Tailnet access, Compose allows the configured Studio hostname (`cortex`) to
reach Vite. If you rename that machine or use a different MagicDNS hostname,
change `ATLAS_UI_ALLOWED_HOSTS` in `docker-compose.yml` (a comma-separated list
is supported) and recreate the frontend service.

Run the API in one terminal:

```bash
uv run atlas
```

Then run the interface in another:

```bash
cd frontend
npm install
npm run dev
```

Open the local URL printed by Vite. The interface defaults to the fast brain;
choose Deep brain before sending a request to use `qwen3.5:122b-a10b`. When the
frontend and API run on different machines, start the interface with
`ATLAS_API_URL=http://<atlas-host>:8000 npm run dev`.

Run the test suite with:

```bash
uv run --all-extras pytest
```

When dependencies change, update the lockfile and commit it with the related
`pyproject.toml` change:

```bash
uv lock
uv sync --all-extras
```

For one-off dependency changes, prefer uv's project commands, for example
`uv add <package>` or `uv add --dev <package>`. The generated `uv.lock` is the
source of truth for reproducible installs.

## Configuration

Copy `.env.example` to `.env` for local development. Environment variables use
the `ATLAS_` prefix—for example, `ATLAS_PORT=8001`. `.env` is deliberately
ignored by Git because it may eventually contain connection details or API
credentials; `.env.example` documents only safe defaults.

## Health endpoints

`GET /health/live` confirms that the HTTP process is running. `GET
/health/ready` confirms that ATLAS has completed its startup lifecycle and is
ready to accept work. Container orchestration and future service dependencies
can use readiness to avoid sending events to a partially initialized system.

## Docker

```bash
cp .env.example .env
docker compose up --build
```

When Ollama runs on the Docker host (as it commonly will on macOS), set
`ATLAS_OLLAMA_BASE_URL=http://host.docker.internal:11434` in `.env`; `localhost`
inside the ATLAS container refers to the container itself. Use `docker compose
down` (or send SIGTERM) for a graceful shutdown. Open `http://localhost:3001`
for the chat interface and `http://localhost:8000` for the API.

## Near-term roadmap

Phase 3 has begun with a typed registry, bounded agent loop, and a read-only
local time tool. Weather is next once a weather provider and location policy
are chosen; notes follow when persistent storage is introduced. Persistent
memory, Obsidian, and MQTT follow after the tool policy boundary is tested.
NATS later becomes the internal event bus. Those adapters will translate native
messages into `AtlasEvent` instead of leaking vendor-specific formats through
the application.
