# Archipelago Gossip Consensus Simulation

A peer-to-peer simulation where 4–6 LLM-driven agents ("factions") negotiate toward political consensus over a gossip protocol. The network layer injects configurable latency and packet loss. The simulation runs Monte Carlo trials across a grid of network and topological conditions and logs all events and outcomes for statistical analysis. A Pygame visualizer renders runs for demonstration.

## Research Questions
- **Primary RQ:** How do latency and packet loss affect time-to-consensus and consensus-failure rate?
- **Secondary RQ:** Does topology (mesh vs. hub-and-spoke) change resilience to the same conditions?

## Tech Stack
- **Language:** Python 3.11+
- **Network / Concurrency:** `asyncio` + `websockets`
- **Agents:** Local LLM via `transformers` / `litellm` + structured output validation with `pydantic`
- **Simulation:** Pure Python asynchronous round runner
- **Analysis:** `pandas`, `scipy.stats` (Two-Way ANOVA, Mann-Whitney U), `seaborn`, `matplotlib`
- **Visualization:** `pygame`

## Setup & Usage
See `SPEC.md` for technical specifications and `roadmap.md` for the phased implementation guide.
