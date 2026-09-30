# Project Spec: Archipelago Gossip Consensus Simulation

## 1. Overview

A peer-to-peer simulation where 4–6 LLM-driven agents ("factions") negotiate toward political consensus over a gossip protocol. The network layer injects configurable latency and packet loss. The simulation runs many trials across a grid of conditions and logs everything for statistical analysis. A Pygame visualizer renders one run at a time for demo purposes.

**Primary RQ:** how do latency and packet loss affect time-to-consensus and consensus-failure rate?
**Secondary RQ:** does topology (mesh vs. hub-and-spoke) change resilience to the same conditions?

Build order: network layer with dummy/scripted agents first → swap in real LLM agents → batch experiment runner → analysis → visualizer last.

---

## 2. Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.11+ | one language across all layers |
| Network / concurrency | `asyncio` + `websockets` | simplest way to get real async P2P behavior with controllable delay/drop, no cluster overhead |
| Agent LLM | local Hugging Face model (e.g. Llama-3-8B-Instruct or Qwen2.5-7B-Instruct), via `transformers` | no API cost, no rate limits across hundreds of runs |
| Structured output | `litellm` (local model routing) + JSON schema validation via `pydantic` | enforce parsable agent actions, needed since raw LLM JSON is unreliable |
| Simulation orchestration | plain Python scripts, no Ray | 4–6 agents doesn't need distributed compute; asyncio is enough and far simpler to debug |
| Logging | `csv` / `pandas` (append-only run logs) | matches analysis stack directly |
| Analysis | `pandas`, `scipy.stats` (ANOVA), `seaborn`/`matplotlib` | standard stats + plotting |
| Visualization | `pygame` | 2D island map, single-run playback from a saved log |
| Config | `pyyaml` (one YAML file per experiment sweep) | keeps conditions declarative, not hardcoded |

---

## 3. Repo Structure

```
archipelago-consensus/
├── config/
│   └── experiment.yaml          # latency/loss/topology grid, run counts, timeouts
├── src/
│   ├── network/
│   │   ├── node.py              # Node class: id, neighbors, inbox/outbox
│   │   ├── gossip.py            # gossip push protocol, round loop
│   │   ├── link.py              # simulated link: injects latency + drop probability
│   │   └── topology.py          # builds mesh / hub-and-spoke adjacency
│   ├── agents/
│   │   ├── llm_agent.py         # wraps local model, builds prompt, calls litellm
│   │   ├── schema.py            # pydantic models for agent action JSON
│   │   ├── archetypes.py        # faction personas (cooperative / ambitious / villain)
│   │   └── dummy_agent.py       # scripted/random agent for testing network layer alone
│   ├── simulation/
│   │   ├── runner.py            # single-run orchestrator: init nodes, drive rounds, detect consensus
│   │   ├── consensus.py         # consensus-check logic (what counts as "unified")
│   │   └── monte_carlo.py       # sweeps config grid, runs N trials per condition, writes logs
│   ├── logging/
│   │   └── run_logger.py        # per-event CSV writer (message sent/dropped/received, consensus checkpoint)
│   └── viz/
│       ├── map_view.py          # Pygame island map + agent nodes
│       └── playback.py          # replays a saved run log frame by frame
├── analysis/
│   ├── load_logs.py             # loads all run CSVs into one dataframe
│   ├── stats.py                 # Two-Way ANOVA, Mann-Whitney for topology
│   └── plots.py                 # convergence curves, failure-rate heatmap
├── data/
│   └── runs/                    # one CSV per run, named by condition + run id
├── requirements.txt
└── README.md
```

---

## 4. Data Flow

```
experiment.yaml
      │
      ▼
monte_carlo.py ──► for each condition (latency, loss, topology):
      │                for each of N trials:
      │                    runner.py:
      │                        1. topology.py builds adjacency for this trial
      │                        2. node.py instantiates one Node per agent
      │                        3. gossip.py drives rounds:
      │                             - each node picks neighbor(s), sends state via link.py
      │                             - link.py sleeps `latency` ms, then drops with `loss` probability
      │                             - receiving node calls llm_agent.py → gets JSON action
      │                             - schema.py validates; on failure, retry once, else log parse-failure
      │                        4. consensus.py checks convergence after each round
      │                        5. run_logger.py writes every event + final result to data/runs/*.csv
      │
      ▼
analysis/load_logs.py → stats.py (ANOVA / Mann-Whitney) → plots.py (figures for the paper)

viz/playback.py reads one saved run CSV independently, for demo purposes — not in the hot path of experiments.
```

---

## 5. Component Details

### 5.1 Network layer (`src/network/`)
- **`Node`**: id, faction name, list of neighbor ids (from topology), inbox queue, current belief state (who it's currently allied with / voting for).
- **`Link`**: wraps a send call — `await asyncio.sleep(latency_ms / 1000)`, then `random.random() < loss_rate` decides drop; if not dropped, delivers to the neighbor's inbox. Every call emits a log event regardless of outcome.
- **`Gossip`**: standard push-gossip round — each node each round picks one (or k) random neighbors and pushes its current state. Round ends when all nodes have had one send opportunity. Runs for up to `max_rounds` (config, e.g. 15) or until `consensus.py` reports convergence.
- **`Topology`**: `build_mesh(n)` → fully connected; `build_hub_spoke(n)` → one designated hub node, all others connect only to it. Returns adjacency dict consumed by `Node` init.

### 5.2 Agent layer (`src/agents/`)
- **`LLMAgent.decide(state, inbox_messages)`**: builds a first-person prompt from the faction's archetype + current known state + incoming gossip, calls the local model via `litellm`, parses the JSON response against `schema.py`.
- **Action schema** (pydantic, strict):
  ```json
  {
    "action": "forge_alliance | break_alliance | hold | broadcast_state",
    "target": "faction_id or null",
    "dialogue": "short first-person string"
  }
  ```
- **Retry policy**: one retry with a "your last output was invalid JSON, respond again in the exact schema" repair prompt; second failure logs a parse-failure event and the node holds its previous state for that round.
- **Archetypes**: 3 presets (cooperative, self-interested, adversarial/"deceiver") as system-prompt fragments; assigned per-faction per trial per the config's archetype ratio.
- **`DummyAgent`**: returns scripted or random valid actions with zero LLM latency — used to validate the network layer end-to-end before the real model is wired in.

### 5.3 Simulation orchestration (`src/simulation/`)
- **`runner.py`**: runs exactly one trial given one fully-specified condition (latency, loss, topology, archetype mix); returns a result dict (converged: bool, rounds_to_consensus, message_count, parse_failures).
- **`consensus.py`**: convergence = all nodes' belief state points to the same outcome (e.g. same allied bloc / same "unified leader"). Checked after every round.
- **`monte_carlo.py`**: reads `experiment.yaml`, expands the condition grid (3 latency × 3 loss [× 2 topology for the secondary RQ subset]), runs `runs_per_condition` trials of each via `runner.py`, writes one row per trial to a master results CSV plus the full per-event logs.

### 5.4 Logging (`src/logging/run_logger.py`)
Two log types per trial:
- **Event log** (one row per message): `trial_id, round, sender, receiver, sent_at, latency_ms, dropped(bool), delivered_at`
- **Trial summary** (one row per trial): `trial_id, latency, loss_rate, topology, archetype_mix, converged, rounds_to_consensus, total_messages, parse_failures`

### 5.5 Analysis (`analysis/`)
- `load_logs.py`: concatenates all trial-summary CSVs into one dataframe.
- `stats.py`: `scipy.stats.f_oneway` / statsmodels Two-Way ANOVA on `rounds_to_consensus ~ latency * loss_rate`; Mann-Whitney U for topology comparison subset.
- `plots.py`: heatmap of failure rate over the latency×loss grid; line plot of time-to-consensus vs. loss rate faceted by latency.

### 5.6 Visualization (`src/viz/`)
- Standalone from the experiment loop — reads one saved event log and replays it: islands as nodes on a 2D map, animated dots for in-flight messages, node color shifts as alliances change, dropped messages flash red. Not required for statistical results, only for the demo/presentation.

---

## 6. Config File Shape (`config/experiment.yaml`)

```yaml
agents: 5
max_rounds: 15
runs_per_condition: 15
model: "meta-llama/Meta-Llama-3-8B-Instruct"

primary_grid:
  latency_ms: [0, 500, 2000]
  loss_rate: [0.0, 0.15, 0.30]
  topology: ["mesh"]

secondary_grid:
  latency_ms: [0, 2000]
  loss_rate: [0.0, 0.30]
  topology: ["mesh", "hub_spoke"]

archetype_mix:
  cooperative: 0.5
  self_interested: 0.3
  adversarial: 0.2
```

---

## 7. Dependencies (`requirements.txt`)

```
websockets>=12.0
transformers>=4.44
torch>=2.3
litellm>=1.40
pydantic>=2.7
pyyaml>=6.0
pandas>=2.2
scipy>=1.13
statsmodels>=0.14
seaborn>=0.13
matplotlib>=3.9
pygame>=2.5
```

---

## 8. Build Order (for the vibe-coding tool)

1. `network/` + `agents/dummy_agent.py` + `simulation/runner.py` for a single trial with dummy agents → confirm gossip rounds, latency injection, and packet drop all work and produce a sane event log.
2. Swap in `agents/llm_agent.py` + `schema.py` — confirm structured JSON output and the retry-once policy work on a handful of manual trials.
3. `simulation/monte_carlo.py` — run the full condition grid at reduced scale first (e.g. 2 runs/condition) to sanity-check timing before committing to the full 15/condition sweep.
4. `analysis/` scripts against the resulting logs.
5. `viz/` last, purely for the presentation demo.
