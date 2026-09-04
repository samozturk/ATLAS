# ATLAS Development Roadmap — Summary

## Goal

Build ATLAS incrementally from a simple local LLM assistant into a reliable, event-driven, proactive home AI system.

---

## Phase 1 — Skeleton
- Python project structure
- Configuration and `.env`
- Structured logging
- Event/message types
- Health checks
- Docker
- Graceful shutdown

**Milestone:** ATLAS starts reliably.

---

## Phase 2 — Local LLM
- Ollama integration
- LLM abstraction/interface
- Streaming responses
- Prompt management
- Timeouts and retries
- Model configuration

**Milestone:** Chat with ATLAS locally.

---

## Phase 3 — Tool Calling
- Standard tool interface
- JSON/input schemas
- Tool execution layer
- Start with:
  - Weather
  - Time
  - Notes search
  - Notes creation

**Milestone:** ATLAS can decide when to call a tool and use its result.

---

## Phase 4 — Persistent Memory
- Conversation storage
- Messages
- Memories
- Events
- Tool-call history
- Structured database
- Add vector/semantic search later if needed

**Milestone:** ATLAS remembers information between sessions.

---

## Phase 5 — Obsidian Integration
- Search notes
- Read notes
- Create notes
- Update notes

**Architecture rule:**
- Database → application state
- Obsidian → human-readable knowledge
- Vector search → semantic retrieval

**Milestone:** ATLAS can use and maintain your Obsidian knowledge base.

---

## Phase 6 — MQTT / Home Devices
- Connect ESP32 devices through MQTT
- Define device/event topics
- MQTT adapter
- Convert MQTT messages into ATLAS events

Example events:
- `temperature.changed`
- `motion.detected`
- `door.opened`
- `device.offline`

**Milestone:** Physical devices can communicate with ATLAS.

---

## Phase 7 — Event-Driven ATLAS
Introduce NATS as the internal event bus.

```text
MQTT / Services → NATS → ATLAS → Reasoning → Action
```

Use deterministic rules for simple events.

Use the LLM only when interpretation/reasoning is actually required.

**Milestone:** ATLAS can react to things happening without being directly asked.

---

## Phase 8 — Voice Interface
- Wake word
- Streaming STT
- Conversation sessions
- Streaming LLM responses
- Streaming TTS
- Interruption/barge-in
- Silence detection

```text
Microphone → Wake Word → STT → ATLAS → LLM/Tools → TTS → Speaker
```

**Milestone:** Talk to ATLAS naturally.

---

## Phase 9 — Context Awareness
Combine:
- Time
- Calendar
- Weather
- User context
- Home sensors
- Devices
- Previous conversations
- Notes

**Milestone:** ATLAS can reason about the current situation rather than isolated requests.

---

## Phase 10 — Planning / Agents
Introduce multi-step task execution.

```text
Request
  ↓
Planner
  ↓
Task Graph
  ↓
Tools
  ↓
Results
  ↓
Validation
  ↓
Response
```

Add:
- Tool permissions
- Iteration limits
- Timeouts
- Approval requirements
- Audit logs

**Milestone:** ATLAS can complete multi-step tasks.

---

## Phase 11 — Autonomous Behaviors
Move from reactive to proactive behavior.

### Reactive
`Event → Rule → Action`

### Intelligent
`Event → Context → LLM reasoning → Action`

### Autonomous
`Goal → Plan → Observe → Act → Evaluate → Replan`

**Milestone:** ATLAS can proactively help without being explicitly asked.

---

## Phase 12 — Security & Permissions
Define tool permission levels:

- READ
- WRITE
- EXECUTE
- DESTRUCTIVE

Define execution policies:

- Automatic
- Confirmation required
- Never automatic

Add complete action auditing.

**Milestone:** ATLAS can act safely.

---

## Phase 13 — Observability
Add:
- Prometheus
- Grafana
- Event traces
- LLM latency
- Token usage
- Tool latency
- Event throughput
- Failed calls
- Agent iterations
- MQTT/NATS activity
- Resource usage

**Milestone:** You can understand what ATLAS is doing internally.

---

## Phase 14 — Reliability
Implement:
- Retries
- Timeouts
- Circuit breakers
- Dead-letter events
- Idempotency
- Correlation IDs
- Persistent event state
- Recovery
- Health checks
- Automatic restarts
- Backups

**Critical rule:** The home should continue functioning when the LLM is unavailable.

**Milestone:** ATLAS becomes dependable rather than experimental.

---

# Full Development Sequence

```text
1.  Skeleton
        ↓
2.  Local LLM
        ↓
3.  Tool Calling
        ↓
4.  Persistent Memory
        ↓
5.  Obsidian
        ↓
6.  MQTT
        ↓
7.  NATS / Event Bus
        ↓
8.  Voice
        ↓
9.  Context Awareness
        ↓
10. Planning / Agents
        ↓
11. Autonomous Behavior
        ↓
12. Permissions / Security
        ↓
13. Observability
        ↓
14. Reliability
        ↓
15. Full ATLAS
```

# ATLAS v1 Definition

The first meaningful ATLAS should be able to:

- Hold conversations
- Use local LLMs through Ollama
- Call tools
- Remember information
- Read/write Obsidian
- Access weather/calendar data
- Communicate with MQTT devices
- React to home events

Everything after that increases autonomy, intelligence, reliability, and sophistication.

# Architecture Principle

**Build ATLAS as a system, not as a giant agent.**

The LLM is one component:

```text
             ┌─────────────┐
             │     LLM     │
             └──────┬──────┘
                    │
┌───────────┐       │       ┌───────────┐
│   Events  │──────ATLAS────│   Tools   │
└───────────┘       │       └───────────┘
                    │
             ┌──────┴──────┐
             │   Memory    │
             └─────────────┘
```

Keep the core independent from any particular LLM, framework, transport, or device ecosystem.
