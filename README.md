# Archipelago Gossip Consensus Simulation

A peer-to-peer simulation where 4–6 AI agents ("factions") negotiate toward political consensus over a push-gossip protocol. The network layer injects configurable latency and packet loss across different topologies (`mesh` vs. `hub_spoke`).

Two agent architectures run on the exact same network and belief dynamics:
- **LLM Agent (`llm`):** A locally hosted open-weight LLM (`ollama/llama3.2:3b`) prompted with faction personas (`cooperative`, `self_interested`, `adversarial`) and constrained to a strict Pydantic JSON action schema with a 1-retry repair policy.
- **Baseline Agent (`baseline`):** A deterministic, non-LLM utility-based agent using DeGroot-style belief updates (`b_i(t+1) = (1 - s_i) * b_i(t) + s_i * sum_j T_ij * b_j(t)`).

---

## Research Questions
- **Primary RQ (Research Methodology):** How do latency and packet loss affect time-to-consensus and consensus-failure rate?
- **Secondary RQ (Research Methodology):** Does network topology (`mesh` vs. `hub_spoke`) change resilience to the same conditions?
- **AI Course Comparison:** How do LLM agents compare with the utility-based baseline agent on correctness, consistency, agreement, and efficiency under the same network conditions?

---

## How Do I Run This? (Step-by-Step Tutorial)

### 1. Prerequisites
1. **Python 3.11+** and **Git** installed.
2. **Ollama** installed (required only when running the `llm` agent arm):
   - Download from [ollama.com/download](https://ollama.com/download).
   - Pull the fixed experiment model (**do not swap models mid-experiment**, as the model is a controlled variable in [`config/experiment.yaml`](config/experiment.yaml)):
     ```bash
     ollama pull llama3.2:3b
     ```
   - Ensure the Ollama application or server (`ollama serve`) is running in the background.

---

### 2. Clone & Set Up the Python Environment

**Windows (PowerShell):**
```powershell
git clone https://github.com/MichaelFirstAC/Archipelago-Gossip-Consensus.git
cd Archipelago-Gossip-Consensus

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**macOS / Linux (Bash/Zsh):**
```bash
git clone https://github.com/MichaelFirstAC/Archipelago-Gossip-Consensus.git
cd Archipelago-Gossip-Consensus

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

### 3. Run the Verification Test Suites

Before running experiments or starting a new phase, verify that all completed phases pass on your machine:

```powershell
# Phase 1: Network layer, topologies, link delay/drop, and dummy agents
python tests/test_phase1.py

# Phase 2A: DeGroot belief math, total-variation consensus score, and baseline utility agent
python tests/test_phase2a.py

# Round Deadline: Stale-message delivery when link latency exceeds the gossip period
python tests/test_round_deadline.py

# Phase 2B (Unit): Prompt building, schema validation, and retry-once-then-hold fallback
python tests/test_phase2b.py

# Phase 2B (Live Ollama): Runs a real 4-agent trial against local ollama/llama3.2:3b
python tests/test_phase2b.py --live
```

---

### 4. Run a Single Simulation Trial from the CLI

You can run individual trials directly via `src.simulation.runner` using any of the three agent types (`baseline`, `llm`, or `dummy`):

**A. Fast Baseline Agent Trial (no LLM needed):**
```powershell
python -m src.simulation.runner --trial-id demo_baseline --agent-type baseline --latency 0 --loss 0.0 --topology mesh --seed 42
```

**B. Real LLM Agent Trial (requires Ollama running `llama3.2:3b`):**
```powershell
python -m src.simulation.runner --trial-id demo_llm --agent-type llm --model ollama/llama3.2:3b --latency 500 --loss 0.15 --topology mesh --seed 42
```

**C. Hub-and-Spoke Topology Trial under Packet Loss:**
```powershell
python -m src.simulation.runner --trial-id demo_hub --agent-type baseline --latency 2000 --loss 0.30 --topology hub_spoke --seed 42
```

**Available CLI Flags (`python -m src.simulation.runner --help`):**
| Flag | Default | Description |
|---|---|---|
| `--trial-id` | `test_trial_001` | Unique identifier for output CSV files |
| `--agent-type` | `baseline` | Agent type: `baseline`, `llm`, or `dummy` |
| `--model` | `ollama/llama3.2:3b` | LiteLLM model string (for `--agent-type llm`) |
| `--temperature` | `0.7` | LLM sampling temperature |
| `--latency` | `0.0` | Simulated link latency in milliseconds |
| `--loss` | `0.0` | Packet drop probability in `[0.0, 1.0]` |
| `--topology` | `mesh` | Network graph: `mesh` or `hub_spoke` |
| `--agents` | `5` | Number of faction nodes |
| `--max-rounds` | `20` | Maximum gossip rounds before timeout |
| `--fanout` | `1` | Random neighbors each node pushes to per round |
| `--round-ms` | `1000.0` | Gossip round window in ms (slower messages arrive late/stale) |
| `--epsilon` | `0.05` | Consensus threshold on max pairwise total-variation distance |
| `--seed` | `None` | Random seed for reproducible trials |
| `--log-dir` | `data/runs` | Output directory for CSV logs |

---

### 5. Run the Phase 3 Batch Experiment Sweeps (`monte_carlo.py`)

`src.simulation.monte_carlo` reads `config/experiment.yaml`, expands the primary (9 `mesh` cells) and secondary (4 `hub_spoke` cells) grids, and is **fully resumable** (it automatically skips already-completed trials if interrupted).

```powershell
# Verify Phase 3 grid expansion & resumability unit tests
python tests/test_phase3.py

# Step 1: Baseline sweep across the primary grid (9 conditions x 30 runs = 270 trials, ~30s)
python -m src.simulation.monte_carlo --agent-type baseline --grid primary

# Step 2: LLM Pilot sweep across all 13 unique conditions (2 runs per condition = 26 trials)
python -m src.simulation.monte_carlo --agent-type llm --grid all --runs 2

# Step 3: Full LLM sweep across all 13 unique conditions (15 runs per condition = 195 trials;
# automatically resumes from the 26 pilot trials and runs only the remaining 169 trials)
python -m src.simulation.monte_carlo --agent-type llm --grid all
```

---

### 6. Where Are the Output Logs?

Every trial writes three CSV artefacts to `data/runs/`:
1. **`data/runs/events_<trial_id>.csv`** — Per-message network log (`trial_id, round, sender, receiver, sent_at, latency_ms, dropped, delivered_at`).
2. **`data/runs/decisions_<trial_id>.csv`** — Per-agent decision & AI telemetry log (`trial_id, round, node, agent_type, valid_json, valid_action, vote, baseline_vote, decision_latency_ms, tokens`).
3. **`data/runs/trials_summary.csv`** — Appended master summary row (`trial_id, agent_type, latency, loss_rate, topology, archetype_mix, converged, rounds_to_consensus, total_messages, parse_failures`).

*(Note: `data/runs/*.csv` is ignored by git so local test runs stay clean. When sharing the final Phase 3 Monte Carlo dataset across teammates, either share the `data/runs/` folder directly or force-add the final CSVs with `git add -f data/runs/*.csv`).*

---

## Project Architecture & Roadmap
- See [`SPEC.md`](SPEC.md) for full architecture, mathematical definitions, and experimental grid specifications.
- See [`roadmap.md`](roadmap.md) for the phase-by-phase implementation checklist (**Phases 0, 1, 2A, and 2B are complete; Phase 3 is next**).
