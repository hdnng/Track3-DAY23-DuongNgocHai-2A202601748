"""Tests for persistence, SQLite checkpointer, state history, and HITL."""

from pathlib import Path

from dotenv import load_dotenv

from langgraph_agent_lab.graph import build_graph
from langgraph_agent_lab.persistence import build_checkpointer
from langgraph_agent_lab.routing import route_after_approval
from langgraph_agent_lab.state import Route, Scenario, initial_state

load_dotenv()



def test_sqlite_checkpointer_creation(tmp_path: Path) -> None:
    db_file = tmp_path / "test_checkpoints.sqlite"
    saver = build_checkpointer(kind="sqlite", database_url=str(db_file))
    assert saver is not None
    assert db_file.exists()


def test_hitl_rejection_routing() -> None:
    state_approved = {"approval": {"approved": True}}
    assert route_after_approval(state_approved) == "tool"

    state_rejected = {"approval": {"approved": False}}
    assert route_after_approval(state_rejected) == "clarify"


def test_state_history_and_persistence(tmp_path: Path) -> None:
    db_file = tmp_path / "test_history.sqlite"

    saver = build_checkpointer(kind="sqlite", database_url=str(db_file))
    graph = build_graph(checkpointer=saver)

    scenario = Scenario(
        id="persist-test",
        query="How do I reset my password?",
        expected_route=Route.SIMPLE,
    )
    state = initial_state(scenario)
    config = {"configurable": {"thread_id": "thread-persist-test"}}

    result = graph.invoke(state, config=config)
    assert result["route"] == "simple"
    assert result.get("final_answer")

    history = list(graph.get_state_history(config))
    assert len(history) >= 2
