"""Lightweight HTTP bridge for the EbbingContext demo.

Provides a REST API that demo.jsx can call to interact with a real MemoryEngine.
Run with: python -m demo.server
"""

from __future__ import annotations

import json
from http.server import HTTPServer, BaseHTTPRequestHandler

from ebbingcontext import MemoryEngine, DecayStrategy
from ebbingcontext.embedding.lite import LiteEmbeddingProvider


engine = MemoryEngine(embedding_provider=LiteEmbeddingProvider())


class DemoHandler(BaseHTTPRequestHandler):
    """Minimal REST handler for the demo frontend."""

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _json_response(self, data: dict | list, status: int = 200) -> None:
        body = json.dumps(data, default=str, ensure_ascii=False).encode()
        self.send_response(status)
        self._cors()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length))

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self) -> None:
        if self.path == "/api/memories":
            agent_id = "demo"
            active = engine.active.get_all(agent_id)
            warm = engine.warm.get_all(agent_id)
            result = []
            for item in active + warm:
                result.append({
                    "id": item.id,
                    "content": item.content,
                    "decay_strategy": item.decay_strategy.value,
                    "sensitivity": item.sensitivity.value,
                    "importance": item.importance,
                    "strength": item.strength,
                    "access_count": item.access_count,
                    "layer": item.layer.value,
                    "created_at": item.created_at,
                })
            self._json_response(result)

        elif self.path == "/api/stats":
            self._json_response({
                "active_count": engine.active.count,
                "warm_count": engine.warm.count,
                "archive_count": engine.archive.count,
                "audit_count": engine.archive.audit_count
                if hasattr(engine.archive, "audit_count")
                else len(engine.archive._audit_log),
            })

        else:
            self._json_response({"error": "not found"}, 404)

    def do_POST(self) -> None:
        body = self._read_body()

        if self.path == "/api/store":
            content = body.get("content", "")
            if not content:
                self._json_response({"error": "content required"}, 400)
                return
            item = engine.store(
                content=content,
                agent_id="demo",
                source_type=body.get("source_type", "user"),
                importance=body.get("importance"),
            )
            self._json_response({
                "id": item.id,
                "content": item.content,
                "decay_strategy": item.decay_strategy.value,
                "importance": item.importance,
                "strength": item.strength,
            })

        elif self.path == "/api/recall":
            query = body.get("query", "")
            results = engine.recall(query=query, agent_id="demo", top_k=body.get("top_k", 10))
            self._json_response([
                {
                    "id": r.item.id,
                    "content": r.item.content,
                    "score": r.final_score,
                    "similarity": r.similarity,
                    "strength": r.strength,
                }
                for r in results
            ])

        elif self.path == "/api/pin":
            memory_id = body.get("memory_id", "")
            try:
                item = engine.pin(memory_id, agent_id="demo")
                self._json_response({"id": item.id, "pinned": True})
            except (KeyError, ValueError) as e:
                self._json_response({"error": str(e)}, 400)

        elif self.path == "/api/forget":
            memory_id = body.get("memory_id", "")
            try:
                item = engine.forget(memory_id)
                self._json_response({"id": item.id, "forgotten": True})
            except KeyError as e:
                self._json_response({"error": str(e)}, 404)

        elif self.path == "/api/migrate":
            count = engine.run_migration(agent_id="demo")
            self._json_response({"migrated": count})

        else:
            self._json_response({"error": "not found"}, 404)

    def log_message(self, format, *args) -> None:
        print(f"[demo] {args[0]}")


def main(port: int = 8765) -> None:
    server = HTTPServer(("localhost", port), DemoHandler)
    print(f"EbbingContext demo server running at http://localhost:{port}")
    print("Endpoints: /api/memories, /api/store, /api/recall, /api/pin, /api/forget, /api/migrate")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
