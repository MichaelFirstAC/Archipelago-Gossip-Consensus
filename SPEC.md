# Project Spec: Archipelago Gossip Consensus Simulation

## 1. Overview

A peer-to-peer simulation where 5 AI agents ("factions", configurable 4–6) negotiate toward political consensus over a gossip protocol. The network layer injects configurable latency and packet loss. The simulation runs many trials across a grid of conditions and logs everything for statistical analysis. A Pygame visualizer is planned to render one run at a time for demo purposes (not built yet).

Two kinds of agent run on the same network, so the AI side can compare approaches:

- **LLM agent**: a locally hosted open-weight LLM (pretrained, prompted, not trained).
- **Baseline agent**: a non-LLM utility-based agent using a DeGroot-style belief update (written by us, no model needed).

**Primary RQ (RM course):** how do latency and packet loss affect time-to-consensus and consensus-failure rate?
**Secondary RQ (RM course):** does topology (mesh vs. hub-and-spoke) change resilience to the same conditions?
**AI-course comparison (extra, not part of the RM RQs):** how do LLM agents compare with the baseline agent on correctness, consistency and efficiency under the same conditions?

---

## 2. Tech Stack

| Layer | Choice | Why |
| --- | --- | --- |
| Language | Python 3.11+ | one language across all layers |
| Network / concurrency | `asyncio` with a simulated in-process link (`asyncio.sleep` delay, Bernoulli drop); no real sockets | controllable, reproducible delay/drop with no cluster overhead |
| LLM agent | one fixed local open-weight model, `ollama/llama3.2:3b`, run through Ollama and called via `litellm` | no API cost or rate limits across hundreds of runs; the model is a control variable, so never swap it mid-experiment |
| Structured output | `litellm` + `pydantic` | enforce parsable agent actions |
| Baseline agent | plain Python + `numpy` | non-LLM comparison arm, fast and deterministic |
| Orchestration | plain Python scripts, no Ray | 5 agents doesn't need distributed compute |
| Logging | `csv` / `pandas` | matches the analysis stack |
| Analysis | `pandas`, `scipy.stats`, `statsmodels`, `seaborn`/`matplotlib` | ANOVA, chi-square, Mann-Whitney, plots |
| Visualization | `pygame-ce` | single-run playback from a saved log (planned) |
| Config | `pyyaml` | declarative experiment grids |

---

## 3. Repo Structure

```
archipelago-gossip-consensus/
├── config/
│   └── experiment.yaml
├── src/
│   ├── network/
│   │   ├── node.py              # Node: id, neighbors, inbox, state
│   │   ├── gossip.py            # round-based push-gossip loop
│   │   ├── link.py              # injects latency + drop probability
│   │   └── topology.py          # mesh / hub-and-spoke adjacency
│   ├── agents/
│   │   ├── base_agent.py        # shared interface: decide(state, inbox) -> Action
│   │   ├── dummy_agent.py       # scripted/random, for network testing
│   │   ├── baseline_agent.py    # utility-based non-LLM agent
│   │   ├── belief.py            # belief update + consensus score (our own AI/math component)
│   │   ├── llm_agent.py         # wraps the local model, builds prompt, calls litellm
│   │   ├── schema.py            # pydantic models for the action JSON
│   │   ├── archetypes.py        # faction personas / utility weights
│   │   └── factory.py           # builds the agent type named in config
│   ├── simulation/
│   │   ├── runner.py            # one trial
│   │   ├── consensus.py         # "unified" check (uses belief.py score)
│   │   └── monte_carlo.py       # sweeps the grid, N trials per condition
│   ├── logging/
│   │   └── run_logger.py
│   └── viz/
│       ├── map_view.py          # placeholder (Phase 5)
│       └── playback.py          # placeholder (Phase 5)
├── analysis/
│   ├── load_logs.py
│   ├── stats.py                 # ANOVA, chi-square, Mann-Whitney, LLM-vs-baseline, AI metrics
│   ├── plots.py
│   └── consistency.py           # decision-consistency measurement (re-queries the LLM)
├── tests/
├── data/runs/
├── results/
│   ├── tables/
│   └── figures/
├── requirements.txt
└── README.md
```

---

## 4. Data Flow

```
experiment.yaml
      │
      ▼
monte_carlo.py ──► for each condition (latency, loss, topology, agent_type):
      │                for each of N trials:
      │                    runner.py:
      │                        1. topology.py builds adjacency
      │                        2. factory.py builds one agent per Node (llm | baseline | dummy)
      │                        3. gossip.py drives rounds:
      │                             - each node pushes its state to neighbors via link.py
      │                             - link.py sleeps `latency` ms, then drops with `loss` probability
      │                             - receiving node's agent.decide() returns an action + vote
      │                             - LLM only: schema.py validates, retry once, else log parse-failure
      │                        4. consensus.py checks convergence after each round
      │                        5. run_logger.py writes events + trial summary
      ▼
analysis/ → stats.py → plots.py → results/
```

---

## 5. Component Details

### 5.1 Network layer (`src/network/`)

- **`Node`**: id, faction name, neighbor ids, inbox queue, current state. State includes a `vote` (the outcome this faction currently prefers, e.g. a candidate unifying leader).
- **`Link`**: `await asyncio.sleep(latency_ms / 1000)`, then drop if `random.random() < loss_rate`; otherwise deliver. Every call is logged either way. The link is simulated in-process; no real sockets are used.
- **`Gossip`**: each round every node pushes its belief vector to one (or k, `fanout`) random neighbors and waits at most `round_ms` (the gossip period) for delivery. A message slower than the period keeps flying and lands in a later round, so it is used late with stale beliefs; this is how latency affects rounds-to-consensus (latencies below the period behave like zero latency). Stops at `max_rounds` or when consensus is reached; messages still in flight are cancelled.
- **`Topology`**: `build_mesh(n)`, `build_hub_spoke(n)`.

### 5.2 Agent layer (`src/agents/`)

All agents implement `decide(state, inbox) -> Action` so the network never knows which kind it is running.

**Action schema** (pydantic, strict):

```json
{
  "action": "forge_alliance | break_alliance | hold | broadcast_state",
  "target": "faction_id or null",
  "vote": "faction_id (preferred unifying leader)",
  "dialogue": "short first-person string"
}
```

**`belief.py` (our own component, shared by the baseline and by the metrics)**

- Each node keeps a belief vector over the K candidate outcomes (probabilities summing to 1).
- **DeGroot-style update:** `b_i(t+1) = (1 − s_i) · b_i(t) + s_i · Σ_j T_ij · b_j(t)`, where `s_i` is the faction's openness (stubbornness = 1 − s_i, set by archetype) and `T_ij` are normalized trust weights over the neighbors whose messages arrived this round. Dropped messages simply don't contribute, which is how network loss reaches the agent. The gossip loop applies this update to every node, whatever its agent type; the agent then chooses its `vote` and `action` from the updated state. This is what makes `baseline_vote` available for the LLM arm.
- **Consensus score:** maximum pairwise total-variation distance between nodes' beliefs (0 = identical). Consensus = all nodes' argmax votes agree and the score is below a threshold `epsilon` (config, 0.05).

**`baseline_agent.py`**

- Utility-based agent: picks the vote and action that maximize expected utility `U = w_gain · expected_gain + w_coop · alliance_benefit + w_trust · trust`, computed from its current belief vector. Utility weights come from the archetype (cooperative / self-interested / adversarial).
- Deterministic given the same state and inbox. No model, near-zero decision latency, so it runs the full grid cheaply.

**`llm_agent.py`**

- Builds a first-person prompt from archetype + current known state + incoming gossip, calls the local model (`ollama/llama3.2:3b`) via `litellm`, validates against `schema.py`.
- **Retry policy:** one retry with a repair prompt; a second failure logs a parse-failure event and the node holds its previous state that round.
- **Efficiency rules:** keep prompts short, cap `max_new_tokens`, fixed temperature (0.7) and seed (42) from config, log tokens and decision latency per call. Inference only: no training or fine-tuning.

**`dummy_agent.py`**: scripted or random valid actions; keep it for network debugging.

**`archetypes.py`**: three presets (cooperative, self-interested, adversarial). Each provides an LLM system-prompt fragment and baseline utility weights plus openness `s_i`.

**`factory.py`**: returns the agent class named by `agent_type` in config.

### 5.3 Simulation orchestration (`src/simulation/`)

- **`runner.py`**: one trial for one fully specified condition; returns a result dict.
- **`consensus.py`**: convergence check using the belief consensus score.
- **`monte_carlo.py`**: expands the grid and runs `runs_per_condition` trials each; fully resumable. The LLM arm runs the RM design (primary grid plus the secondary topology subset: 13 conditions, 195 trials). The baseline arm runs the primary grid with 30 runs per condition (9 conditions, 270 trials).

### 5.4 Logging (`src/logging/run_logger.py`)

- **Event log** (one row per message): `trial_id, round, sender, receiver, sent_at, latency_ms, dropped, delivered_at`
- **Decision log** (one row per agent decision): `trial_id, round, node, agent_type, valid_json, valid_action, vote, baseline_vote, decision_latency_ms, tokens`
- **Trial summary** (one row per trial): `trial_id, agent_type, latency, loss_rate, topology, archetype_mix, converged, rounds_to_consensus, total_messages, parse_failures`

`baseline_vote` is what the baseline agent would have chosen from the same state; it lets us measure LLM–baseline agreement without a second run.

### 5.5 Analysis (`analysis/`)

- `load_logs.py`: concatenates trial summaries and decision logs.
- `stats.py`:
  - RM design: Two-Way ANOVA on `rounds_to_consensus ~ latency * loss_rate` (LLM mesh, converged runs only); chi-square test of each factor on failure rate; Mann-Whitney U and odds ratio for topology.
  - AI comparison: LLM vs baseline on failure rate (odds ratio) and rounds to consensus (Mann-Whitney U), plus the AI metrics below.
  - Effect sizes: partial η² (ANOVA), Cramér's V (chi-square), rank-biserial correlation (Mann-Whitney U).
- `plots.py`: failure-rate heatmap (latency × loss), time-to-consensus vs. loss faceted by latency, topology comparison, LLM-vs-baseline bars.
- `consistency.py`: re-queries the LLM on the same state to measure decision consistency.
- All outputs are saved to `results/tables/` and `results/figures/`.

**AI metrics (reported alongside the network metrics)**

| Metric | Definition |
| --- | --- |
| JSON validity rate | share of LLM outputs that parse against the schema (first attempt) |
| Action validity rate | share of parsed actions that are legal in the current state |
| LLM–baseline agreement | share of decisions where the LLM's `vote` equals `baseline_vote` |
| Decision consistency | share of identical votes when the same state is re-queried several times |
| Decision latency / tokens | mean time and tokens per decision (efficiency) |

### 5.6 Visualization (`src/viz/`)

Standalone replay of one saved event log: islands as nodes, animated in-flight messages, color shifts as votes change, dropped messages flash red. Demo only. **Not built yet; `map_view.py` and `playback.py` are placeholders.**

### 5.7 Known limitations

- Latency acts through the gossip period: with `round_ms: 1000`, 0 ms and 500 ms behave identically for every agent type.
- The ANOVA is run on converged runs only, so it excludes failed runs.
- The LLM arm fails about 20% of runs even at 0 ms latency and 0% loss.
- The baseline arm was not run on hub-and-spoke.
- Results come from a single small model (Llama 3.2 3B).

---

## 6. Config File Shape (`config/experiment.yaml`)

```yaml
agents: 5
max_rounds: 20
fanout: 1                       # random neighbors each node pushes to per round
round_ms: 1000                  # gossip period; slower messages arrive in later rounds
runs_per_condition: 15          # LLM arm
baseline_runs_per_condition: 30 # baseline arm (cheap)
agent_type: "llm"               # llm | baseline | dummy
model: "ollama/llama3.2:3b"
temperature: 0.7
seed: 42
epsilon: 0.05                   # consensus threshold on belief distance

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
litellm>=1.40
pydantic>=2.7
pyyaml>=6.0
numpy>=1.26
pandas>=2.2
scipy>=1.13
statsmodels>=0.14
seaborn>=0.13
matplotlib>=3.9
pygame-ce>=2.5
```

---

## 8. Course Alignment (for the AI + DS report)

| Part of the system | Course it supports | What to explain in the report |
| --- | --- | --- |
| LLM agents (pretrained model + prompting, structured output) | AI: NLP / deep learning | why a pretrained LLM, how outputs are validated |
| Baseline utility agent + DeGroot belief update | AI: agent design, math background | the utility function and update rule, with a worked example |
| LLM vs baseline comparison and AI metrics | AI: comparing approaches, correctness, efficiency | results tables and discussion |
| Gossip protocol, fault injection, topology | Distributed Systems | protocol design, failure model, why mesh vs hub-and-spoke |

The RM report covers the research design only (RQs, hypotheses, variables, ethics, references).

---

## 9. Build Order

See `roadmap.md`. Short version: network layer with dummy agents (done) → baseline agent and belief module (done) → LLM agent (done) → batch runs for both arms (done) → analysis (done) → visualizer (next) → report and presentation (in progress).
