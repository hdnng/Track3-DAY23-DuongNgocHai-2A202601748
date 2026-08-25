"""Interactive Demo Server for LangGraph Agentic Lab.

Serves the interactive web UI and provides REST API endpoints to run scenarios,
custom queries, inspect live state, and retrieve historical metrics.
"""

from __future__ import annotations

import json
import os
import sys
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Load environment
load_dotenv()

# Add workspace to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from langgraph_agent_lab.graph import build_graph
from langgraph_agent_lab.persistence import build_checkpointer
from langgraph_agent_lab.state import Route, Scenario, initial_state


class AgentDemoHandler(SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(200)
        self.end_headers()

    def do_GET(self) -> None:
        url_path = self.path.split("?")[0]

        if url_path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            ui_path = Path(__file__).parent / "demo_ui.html"
            self.wfile.write(ui_path.read_bytes())
            return

        if url_path == "/docs" or url_path == "/lab_explanation.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            doc_path = Path(__file__).parent / "docs" / "LAB_EXPLANATION.html"
            self.wfile.write(doc_path.read_bytes())
            return

        if url_path == "/api/metrics":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            metrics_path = Path(__file__).parent / "outputs" / "metrics.json"
            if metrics_path.exists():
                self.wfile.write(metrics_path.read_bytes())
            else:
                self.wfile.write(b'{"total_scenarios": 0, "scenario_metrics": []}')
            return

        if url_path == "/api/scenarios":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            scenarios = [
                {
                    "id": "S01_simple",
                    "query": "How do I reset my password?",
                    "expected_route": "simple",
                    "requires_approval": False,
                    "description": "General FAQ inquiry (No tools required)",
                },
                {
                    "id": "S02_tool",
                    "query": "Please lookup order status for order 12345",
                    "expected_route": "tool",
                    "requires_approval": False,
                    "description": "Information lookup via external tool",
                },
                {
                    "id": "S03_missing",
                    "query": "Can you fix it?",
                    "expected_route": "missing_info",
                    "requires_approval": False,
                    "description": "Ambiguous/incomplete query requiring clarification",
                },
                {
                    "id": "S04_risky",
                    "query": "Refund this customer and send confirmation email",
                    "expected_route": "risky",
                    "requires_approval": True,
                    "description": "Financial operation requiring Human-In-The-Loop approval",
                },
                {
                    "id": "S05_error",
                    "query": "Timeout failure while processing request",
                    "expected_route": "error",
                    "requires_approval": False,
                    "description": "Transient failure testing cyclic retry loop",
                },
                {
                    "id": "S06_delete",
                    "query": "Delete customer account after support verification",
                    "expected_route": "risky",
                    "requires_approval": True,
                    "description": "High-risk data deletion requiring HITL authorization",
                },
                {
                    "id": "S07_dead_letter",
                    "query": "System failure cannot recover after multiple attempts",
                    "expected_route": "error",
                    "requires_approval": False,
                    "max_attempts": 1,
                    "description": "Exhausted retries escalating to Dead Letter Queue",
                },
            ]
            self.wfile.write(json.dumps(scenarios, indent=2).encode("utf-8"))
            return

        super().do_GET()

    def do_POST(self) -> None:
        url_path = self.path.split("?")[0]

        if url_path == "/api/run":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode("utf-8"))

            scenario_id = data.get("scenario_id", "custom_run")
            query = data.get("query", "")
            expected_route = data.get("expected_route", "simple")
            requires_approval = bool(data.get("requires_approval", False))
            max_attempts = int(data.get("max_attempts", 3))
            approval_decision = data.get("approval_choice", "approved")  # 'approved' or 'rejected'

            checkpointer = build_checkpointer(kind="memory")
            graph = build_graph(checkpointer=checkpointer)

            route_enum = Route(expected_route) if expected_route in [r.value for r in Route] else Route.SIMPLE
            scenario = Scenario(
                id=scenario_id,
                query=query,
                expected_route=route_enum,
                requires_approval=requires_approval,
            )
            state = initial_state(scenario)
            state["max_attempts"] = max_attempts

            # Set explicit human approval decision for HITL demo
            if approval_decision == "rejected":
                state["approval"] = {
                    "approved": False,
                    "reviewer": "demo_supervisor",
                    "comment": "Từ chối bởi Supervisor trên giao diện Web Demo",
                }
            else:
                state["approval"] = {
                    "approved": True,
                    "reviewer": "demo_supervisor",
                    "comment": "Phê duyệt bởi Supervisor trên giao diện Web Demo",
                }

            thread_id = f"demo-{scenario_id}-{int(time.time() * 1000)}"

            config = {"configurable": {"thread_id": thread_id}}

            step_trace: list[dict[str, Any]] = []
            t0 = time.perf_counter()

            try:
                # Stream updates to capture node execution step-by-step
                for step_chunk in graph.stream(state, config=config, stream_mode="updates"):
                    for node_name, node_update in step_chunk.items():
                        step_trace.append({
                            "node": node_name,
                            "timestamp": time.time(),
                            "update": {k: v for k, v in node_update.items() if k not in ("events",)},
                        })

                final_state = graph.get_state(config).values
                latency_ms = int((time.perf_counter() - t0) * 1000)

                response_payload = {
                    "success": True,
                    "latency_ms": latency_ms,
                    "thread_id": thread_id,
                    "step_trace": step_trace,
                    "final_state": {
                        "query": final_state.get("query"),
                        "route": final_state.get("route"),
                        "risk_level": final_state.get("risk_level"),
                        "attempt": final_state.get("attempt"),
                        "max_attempts": final_state.get("max_attempts"),
                        "final_answer": final_state.get("final_answer"),
                        "pending_question": final_state.get("pending_question"),
                        "proposed_action": final_state.get("proposed_action"),
                        "approval": final_state.get("approval"),
                        "messages": final_state.get("messages", []),
                        "tool_results": final_state.get("tool_results", []),
                        "errors": final_state.get("errors", []),
                        "events": final_state.get("events", []),
                    },
                }
            except Exception as exc:
                latency_ms = int((time.perf_counter() - t0) * 1000)
                response_payload = {
                    "success": False,
                    "error": str(exc),
                    "latency_ms": latency_ms,
                    "step_trace": step_trace,
                }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response_payload, indent=2, ensure_ascii=False).encode("utf-8"))
            return

        self.send_error(404, "Endpoint not found")


def run_server(port: int = 8000) -> None:
    server_address = ("", port)
    httpd = HTTPServer(server_address, AgentDemoHandler)
    print(f"============================================================")
    print(f"  LangGraph Agentic Orchestration — Live Interactive Demo  ")
    print(f"============================================================")
    print(f"  * UI Web App:   http://localhost:{port}")
    print(f"  * Lab Docs:     http://localhost:{port}/docs")
    print(f"  * Press Ctrl+C to stop the server.")
    print(f"============================================================")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer shutting down.")
        httpd.server_close()


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    run_server(port)
