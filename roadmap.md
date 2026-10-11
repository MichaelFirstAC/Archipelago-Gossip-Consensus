# Roadmap: Archipelago Gossip Consensus Simulation

Companion to `SPEC.md`. Work through phases in order — do not start a phase until the previous one's "Done when" criteria are met. Each mission/task given to the agent should target exactly one phase (or one sub-item within it), not several at once.

---

## Phase 0 — Project Scaffolding
- [x] Create repo structure exactly as laid out in `SPEC.md` §3
- [x] `requirements.txt` populated from `SPEC.md` §7, virtual env created, deps installed
- [x] Empty `config/experiment.yaml` created with the shape from `SPEC.md` §6 (values can be placeholders for now)

**Done when:** `pip install -r requirements.txt` succeeds and every file/folder in §3 exists (even if empty).

---

## Phase 1 — Network Layer (dummy agents only)
- [x] `network/node.py`: `Node` class (id, neighbors, inbox, state)
- [x] `network/topology.py`: `build_mesh(n)` and `build_hub_spoke(n)`
- [x] `network/link.py`: async send with latency sleep + probabilistic drop, emits a log event either way
- [x] `network/gossip.py`: push-gossip round loop, respects `max_rounds`
- [x] `agents/dummy_agent.py`: returns a scripted or random valid action instantly (no LLM call)
- [x] `simulation/consensus.py`: convergence check
- [x] `simulation/runner.py`: runs one full trial end-to-end using dummy agents
- [x] `logging/run_logger.py`: writes the event log and trial-summary log for one trial

**Done when:** running one trial at `latency=0, loss=0` with dummy agents reaches consensus every time, and running it at `loss=1.0` never reaches consensus (sanity-checks that the drop mechanism actually blocks convergence). Event log and trial-summary CSV both produced and readable.

---

## Phase 2 — Agent Layer (real LLM)
- [ ] `agents/schema.py`: pydantic model for the action JSON (`action`, `target`, `dialogue`)
- [ ] `agents/archetypes.py`: cooperative / self-interested / adversarial system-prompt fragments
- [ ] `agents/llm_agent.py`: loads local model once, builds prompt from state + archetype, calls via `litellm`, validates against schema, retry-once-then-fail policy
- [ ] Swap `dummy_agent.py` for `llm_agent.py` in `runner.py` behind a config flag (keep dummy mode available for future debugging)

**Done when:** a single manual trial with 4–6 real LLM agents produces valid, schema-passing JSON actions on most turns, and a forced-invalid-output test confirms the retry-once-then-log-failure path actually triggers instead of crashing the run.

---

## Phase 3 — Batch Experiment Runner
- [ ] `simulation/monte_carlo.py`: reads `config/experiment.yaml`, expands the primary grid (latency × loss) and secondary grid (topology subset), runs `runs_per_condition` trials per cell
- [ ] Confirm resumability: if the run is interrupted, already-completed trials aren't silently overwritten or re-run
- [ ] **Pilot sweep first:** run the full grid at `runs_per_condition: 2` to check total runtime and catch crashes before committing to the real run
- [ ] **Full sweep:** run at the real `runs_per_condition` (e.g. 15) from `config/experiment.yaml`

**Done when:** `data/runs/` contains one trial-summary row and one event-log file per trial across every condition in both grids, with no crashed/missing cells.

---

## Phase 4 — Analysis
- [ ] `analysis/load_logs.py`: concatenates all trial-summary CSVs into one dataframe
- [ ] `analysis/stats.py`: Two-Way ANOVA (latency × loss on time-to-consensus and on failure rate) + Mann-Whitney U for the topology comparison
- [ ] `analysis/plots.py`: failure-rate heatmap (latency × loss), time-to-consensus line plot faceted by latency, topology comparison plot
- [ ] Parse/action-failure rate reported descriptively alongside the main results

**Done when:** all figures needed for the paper's Results section are generated from `data/runs/` with one script run, and the ANOVA output (F-statistics, p-values) is saved somewhere reusable (not just printed to console).

---

## Phase 5 — Visualization (demo only, not required for results)
- [ ] `viz/map_view.py`: static island layout rendering for N agents
- [ ] `viz/playback.py`: reads one saved event log, animates messages between islands frame by frame, flashes dropped messages, updates node color on state change

**Done when:** a single saved run from Phase 1 or 2 can be replayed visually from start to consensus (or timeout).

---

## Notes for whoever (or whatever agent) is driving this
- Phases 1 and 2 are the highest-risk phases — don't let the agent skip ahead to Phase 3 until Phase 2's manual validation is actually confirmed, since a flaky LLM-JSON pipeline will silently corrupt the whole Monte Carlo dataset if it's not caught first.
- Phase 3's pilot sweep exists specifically to catch runtime/compute problems before burning hours on the full run — don't skip it even if Phase 2 looked solid in isolation.
- Keep `dummy_agent.py` in the codebase after Phase 2 — it's the fastest way to isolate "is this a network bug or an LLM bug" if something breaks later.
