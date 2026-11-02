# Roadmap: Archipelago Gossip Consensus Simulation

Companion to `SPEC.md`. Work through phases in order. Do not start a phase until the previous one's "Done when" criteria are met. Each mission given to the agent should target exactly one phase (or one sub-phase), not several at once.

**Status:** Phase 0, Phase 1, Phase 2A, and Phase 2B are complete. Next up: Phase 3.

---

## Phase 0: Project Scaffolding (DONE)
- [x] Create repo structure as laid out in `SPEC.md` §3
- [x] `requirements.txt` populated, virtual env created, deps installed
- [x] `config/experiment.yaml` created with the shape from `SPEC.md` §6

---

## Phase 1: Network Layer, dummy agents only (DONE)
- [x] `network/node.py`, `network/topology.py`, `network/link.py`, `network/gossip.py`
- [x] `agents/dummy_agent.py`
- [x] `simulation/consensus.py`, `simulation/runner.py`
- [x] `logging/run_logger.py` (event log and trial summary)

**Done when (already met):** at `latency=0, loss=0` dummy agents always reach consensus; at `loss=1.0` they never do; both log files are produced.

---

## Phase 2A: Baseline Agent and Belief Module (non-LLM)
Do this before the LLM. It needs no model, so it is fast to test, and it gives us a trusted reference for the LLM later.

Small changes to Phase 1 code first:
- [x] Add a `vote` field to node state if it is not there (see `SPEC.md` §5.1)
- [x] Add `agents/base_agent.py` with the shared `decide(state, inbox) -> Action` interface; make `dummy_agent.py` implement it
- [x] Add `agents/schema.py` with the action schema (now including `vote`)

New components:
- [x] `agents/belief.py`: belief vector per node, DeGroot-style update with trust weights and openness, consensus score (max pairwise total-variation distance)
- [x] `agents/archetypes.py`: three presets with baseline utility weights and openness (LLM prompt fragments can be filled in Phase 2B)
- [x] `agents/baseline_agent.py`: utility-based agent that picks `vote` and `action` from its belief vector
- [x] `agents/factory.py`: builds the agent named by `agent_type` in config
- [x] Update `simulation/consensus.py` to use the belief consensus score with `epsilon` from config
- [x] Add the decision log (`SPEC.md` §5.4) to `run_logger.py`, including `baseline_vote`

**Done when:**
- unit tests show the belief update behaves correctly on a tiny hand-checkable example (e.g. 3 nodes, known trust matrix, one update step matches the hand calculation)
- a baseline-agent trial at `latency=0, loss=0` reaches consensus reliably, and at `loss=1.0` never does
- the decision log and trial summary are written with the new columns

---

## Phase 2B: LLM Agent (DONE)
- [x] `agents/llm_agent.py`: loads the local model once, builds a first-person prompt from archetype + state + inbox, calls it via `litellm`, validates against `schema.py`
- [x] Retry-once-then-log-failure policy
- [x] Efficiency rules: short prompts, `max_new_tokens` cap, fixed temperature and seed, log tokens and decision latency per call
- [x] Fill in the LLM prompt fragments in `archetypes.py`
- [x] Register the LLM agent in `factory.py` behind `agent_type: llm` (keep dummy and baseline available)

**Done when:** a manual trial with 4–6 LLM agents produces schema-valid actions on most turns, and a forced-invalid-output test confirms the retry path logs a failure instead of crashing. Fix the model choice here and do not change it afterward.

---

## Phase 3: Batch Experiment Runner
- [x] `simulation/monte_carlo.py`: reads `config/experiment.yaml`, expands the primary and secondary grids, runs the right number of trials per cell for each agent type
- [x] Resumability: an interrupted sweep does not overwrite or repeat finished trials
- [ ] **Pilot sweep first:** full grid at `runs_per_condition: 2` for the LLM arm, to check runtime and catch crashes
- [ ] **Baseline sweep:** primary grid at `baseline_runs_per_condition` (cheap, can run first)
- [ ] **Full LLM sweep:** primary grid, then the secondary topology subset, at the real `runs_per_condition`

**Done when:** `data/runs/` has one trial-summary row per trial across every condition and both agent types, plus decision logs, with no crashed or missing cells.

---

## Phase 4: Analysis
- [ ] `analysis/load_logs.py`: loads trial summaries and decision logs
- [ ] `analysis/stats.py`: Two-Way ANOVA (latency × loss), failure-rate analysis, Mann-Whitney U for topology (RM design)
- [ ] AI comparison in `stats.py`: LLM vs baseline on convergence rate and rounds to consensus
- [ ] AI metrics: JSON validity, action validity, LLM–baseline agreement, decision consistency, decision latency and tokens (`SPEC.md` §5.5)
- [ ] `analysis/plots.py`: failure-rate heatmap, time-to-consensus plots, topology plot, LLM-vs-baseline plot
- [ ] Save statistical outputs (F-statistics, p-values) to files, not just the console

**Done when:** every figure and table needed for the Results section is produced by one script run from `data/runs/`.

---

## Phase 5: Visualization (demo only)
- [ ] `viz/map_view.py` and `viz/playback.py`

**Done when:** one saved run replays visually from start to consensus (or timeout).

---

## Phase 6: Reports and Presentation
- [ ] **AI + DS report** (max 2 members on the AI side): background, methodology, implementation (LLM agent, baseline agent and belief math, gossip and fault injection), results (network metrics, AI metrics, LLM vs baseline), ethical considerations
- [ ] **RM report**: research design only (RQs, hypotheses, variables, ethics, references)
- [ ] Presentation and live demo
- [ ] Clean the code (readable, no dead code) and write the README, since implementation efficiency and clarity are graded

---

## Notes for whoever (or whatever agent) is driving this
- Phase 2A before 2B: the baseline is the fast, deterministic reference. If the LLM later behaves oddly, compare it to the baseline to find out whether the network or the model is at fault.
- Do not let the agent skip to Phase 3 until 2B's manual validation is confirmed. A flaky LLM-JSON pipeline silently corrupts the whole dataset.
- The baseline arm is an AI-course comparison, not part of the RM research questions. Keep the RM analysis (ANOVA, topology) on the LLM arm so the RM report matches its proposal.
- Keep `dummy_agent.py` after Phase 2. It is the fastest way to isolate network bugs.
- Expect a small persona floor: with the default archetype mix the baseline fails to unify in about 3% of zero-loss mesh trials (a self-interested or adversarial faction holds out). Treat the `latency=0, loss=0` cell as the control and report effects relative to it.
- Latency acts through the gossip period (`round_ms`): a message slower than one period is applied in a later round with stale beliefs. With `round_ms: 1000`, 0 ms and 500 ms behave the same; 2000 ms is the level that hurts. Lower `round_ms` makes 500 ms matter too but pushes 2000 ms to near-total failure within 20 rounds.
- Hub-and-spoke converges much more slowly than mesh because spokes only hear from the hub (about one message per four rounds). Expect it to sit near the `max_rounds` ceiling under loss; analyze it with rounds-to-consensus and failure rate, not rounds alone.
