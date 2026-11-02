"""Automated verification suite for Phase 2B: LLM Agent, retry policy, and live Ollama trial."""

import argparse
import asyncio
import csv
import shutil
import socket
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.archetypes import ARCHETYPES
from src.agents.baseline_agent import BaselineAgent
from src.agents.dummy_agent import DummyAgent
from src.agents.factory import create_agent
from src.agents.llm_agent import DEFAULT_MODEL, LLMAgent, _LOADED_MODELS
from src.logging.run_logger import RunLogger
from src.network.gossip import GossipProtocol
from src.network.link import Link
from src.network.node import Node
from src.network.topology import build_mesh
from src.simulation.runner import run_trial

LOG_DIR = "data/runs_test_2b"


def _fake_response(content: str, prompt_tokens: int = 40, completion_tokens: int = 20) -> Any:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


def test_archetype_prompts_and_factory():
    print("[TEST] Verifying archetype LLM prompt fragments and factory registration...")
    for name, arch in ARCHETYPES.items():
        assert arch.llm_prompt and len(arch.llm_prompt) > 20, f"Missing llm_prompt on {name}"

    cands = ["faction_0", "faction_1", "faction_2"]
    llm_agent = create_agent("llm", "faction_0", cands, ARCHETYPES["cooperative"])
    base_agent = create_agent("baseline", "faction_0", cands, ARCHETYPES["cooperative"])
    dummy_agent = create_agent("dummy", "faction_0", cands, ARCHETYPES["cooperative"])

    assert isinstance(llm_agent, LLMAgent)
    assert isinstance(base_agent, BaselineAgent)
    assert isinstance(dummy_agent, DummyAgent)
    assert DEFAULT_MODEL in _LOADED_MODELS
    print("  -> OK")


def test_prompt_and_retry_once_policy():
    print("[TEST] Verifying prompt building, retry-once repair, and forced-invalid fallback...")
    cands = ["faction_0", "faction_1", "faction_2"]
    state = {
        "node_id": "faction_0",
        "vote": "faction_0",
        "belief": "faction_1",
        "belief_vector": [0.2, 0.6, 0.2],
        "neighbors": ["faction_1", "faction_2"],
        "candidates": cands,
    }
    inbox = [
        {"sender": "faction_1", "payload": {"belief_vector": [0.1, 0.8, 0.1]}},
    ]

    # Case 1: First attempt succeeds
    calls_1: List[Any] = []

    def ok_completion(**kwargs):
        calls_1.append(kwargs)
        return _fake_response(
            '{"action": "forge_alliance", "target": "faction_1", "vote": "faction_1", "dialogue": "I back faction_1."}'
        )

    agent_ok = LLMAgent(
        "faction_0", cands, ARCHETYPES["cooperative"], completion_fn=ok_completion
    )
    act_1 = agent_ok.decide(state, inbox)
    assert len(calls_1) == 1
    assert act_1.action == "forge_alliance" and act_1.vote == "faction_1"
    assert agent_ok.last_valid_json is True
    assert agent_ok.last_parse_failed is False
    assert agent_ok.last_tokens == 60
    assert agent_ok.last_latency_ms >= 0.0

    # Case 2: First attempt fails, repair retry succeeds
    calls_2: List[Any] = []

    def repair_completion(**kwargs):
        calls_2.append(kwargs)
        if len(calls_2) == 1:
            return _fake_response("NOT VALID JSON AT ALL")
        return _fake_response(
            '{"action": "broadcast_state", "target": null, "vote": "faction_1", "dialogue": "Repaired JSON."}'
        )

    agent_repair = LLMAgent(
        "faction_0", cands, ARCHETYPES["self_interested"], completion_fn=repair_completion
    )
    act_2 = agent_repair.decide(state, inbox)
    assert len(calls_2) == 2, f"Expected 2 calls (1 retry), got {len(calls_2)}"
    assert act_2.action == "broadcast_state" and act_2.vote == "faction_1"
    assert agent_repair.last_valid_json is False, "First attempt was invalid so last_valid_json must be False"
    assert agent_repair.last_parse_failed is False, "Second attempt succeeded so parse_failed must be False"
    assert agent_repair.parse_failures == 0
    assert agent_repair.last_tokens == 120

    # Case 3: Forced invalid output on both attempts -> logs failure, holds previous state, never crashes
    calls_3: List[Any] = []

    def always_bad_completion(**kwargs):
        calls_3.append(kwargs)
        return _fake_response("{broken json: true")

    agent_fail = LLMAgent(
        "faction_0", cands, ARCHETYPES["adversarial"], completion_fn=always_bad_completion
    )
    act_3 = agent_fail.decide(state, inbox)
    assert len(calls_3) == 2, f"Expected 2 calls before fallback, got {len(calls_3)}"
    assert act_3.action == "hold" and act_3.vote == "faction_0" and act_3.target is None
    assert agent_fail.last_valid_json is False
    assert agent_fail.last_parse_failed is True
    assert agent_fail.parse_failures == 1
    print("  -> OK")


async def test_forced_invalid_trial_logs_failure_without_crashing():
    print("[TEST] Verifying forced-invalid output across a full gossip round logs parse_failures...")
    if Path(LOG_DIR).exists():
        shutil.rmtree(LOG_DIR, ignore_errors=True)

    cands = [f"faction_{i}" for i in range(4)]
    adj = build_mesh(cands)

    def bad_completion(**kwargs):
        return _fake_response("<<<unparseable output>>>")

    import numpy as np

    nodes = [
        Node(
            node_id=cid,
            neighbors=adj[cid],
            initial_belief=cid,
            agent=LLMAgent(cid, cands, ARCHETYPES["cooperative"], completion_fn=bad_completion),
            belief_vector=np.full(len(cands), 1.0 / len(cands)),
            archetype=ARCHETYPES["cooperative"],
        )
        for cid in cands
    ]

    logger = RunLogger("forced_invalid_trial")
    link = Link(latency_ms=0.0, loss_rate=0.0, logger=logger)
    protocol = GossipProtocol(nodes=nodes, link=link, logger=logger, agent_type="llm")
    result = await protocol.run(max_rounds=2)

    # 4 nodes * 2 rounds = 8 decisions, all failing both attempts
    assert result["parse_failures"] == 8, f"Expected 8 parse failures, got {result['parse_failures']}"
    assert len(logger.decisions) == 8
    assert all(d["valid_json"] is False for d in logger.decisions)
    print("  -> OK")


def _ollama_reachable(host: str = "127.0.0.1", port: int = 11434) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1.5):
            return True
    except OSError:
        return False


async def test_live_ollama_trial(num_agents: int = 4, max_rounds: int = 2):
    print(f"[TEST] Running live manual trial with {num_agents} LLM agents ({DEFAULT_MODEL})...")
    if Path(LOG_DIR).exists():
        shutil.rmtree(LOG_DIR, ignore_errors=True)

    summary = await run_trial(
        trial_id="live_llm_manual",
        latency_ms=0.0,
        loss_rate=0.0,
        topology="mesh",
        num_agents=num_agents,
        max_rounds=max_rounds,
        agent_type="llm",
        model=DEFAULT_MODEL,
        seed=42,
        log_dir=LOG_DIR,
        save_logs=True,
    )
    print(f"  Trial summary: {summary}")

    decisions_csv = Path(LOG_DIR) / "decisions_live_llm_manual.csv"
    assert decisions_csv.exists(), "Decision log CSV was not created"
    with open(decisions_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) > 0, "No decision rows logged"
    valid_json_count = sum(1 for r in rows if r["valid_json"] == "True")
    valid_action_count = sum(1 for r in rows if r["valid_action"] == "True")
    json_rate = valid_json_count / len(rows)
    action_rate = valid_action_count / len(rows)
    mean_latency = sum(float(r["decision_latency_ms"]) for r in rows) / len(rows)
    mean_tokens = sum(int(r["tokens"]) for r in rows) / len(rows)

    print(
        f"  Decisions: {len(rows)} | JSON valid (1st try): {valid_json_count}/{len(rows)} ({json_rate:.0%}) | "
        f"Action valid: {valid_action_count}/{len(rows)} ({action_rate:.0%}) | "
        f"Mean latency: {mean_latency:.0f} ms | Mean tokens: {mean_tokens:.1f}"
    )
    assert json_rate >= 0.75, f"JSON validity rate too low: {json_rate:.2f}"
    assert action_rate >= 0.75, f"Action validity rate too low: {action_rate:.2f}"
    assert all(int(r["tokens"]) > 0 for r in rows), "Token counts were not recorded"

    shutil.rmtree(LOG_DIR, ignore_errors=True)
    print("  -> Live Ollama manual trial verified!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Run live manual trial against Ollama")
    args = parser.parse_args()

    test_archetype_prompts_and_factory()
    test_prompt_and_retry_once_policy()
    asyncio.run(test_forced_invalid_trial_logs_failure_without_crashing())

    if args.live:
        if not _ollama_reachable():
            raise RuntimeError("Ollama server at 127.0.0.1:11434 is not reachable!")
        asyncio.run(test_live_ollama_trial())
    else:
        print("[INFO] Pass --live to run the live Ollama manual trial.")

    shutil.rmtree(LOG_DIR, ignore_errors=True)
    print("\nALL PHASE 2B TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    main()

