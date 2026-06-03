"""Minimal OpenAI-compatible streaming mock server for offline harness tests.

Streams chunked SSE with configurable simulated TTFT and per-token delay. In
`oracle` mode it parses the queried key from the prompt and returns the correct
magic number, so the quality guardrail can be validated end-to-end.

Run:  python -m opendc_bench.mockserver --port 8000 --ttft 0.05 --tpot 0.005
"""
from __future__ import annotations

import argparse
import json
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CFG = {"ttft": 0.05, "tpot": 0.005, "n_tokens": 16, "oracle": True}


def _oracle_answer(prompt: str) -> str:
    m = re.search(r"magic number for (\w+)\?", prompt)
    if not m:
        return "0"
    key = m.group(1)
    hit = re.search(rf"for {re.escape(key)} is: (\d+)", prompt)
    return hit.group(1) if hit else "0"


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # silence
        pass

    def _chunk(self, data: bytes):
        self.wfile.write(f"{len(data):x}\r\n".encode())
        self.wfile.write(data)
        self.wfile.write(b"\r\n")
        self.wfile.flush()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(length) or b"{}")
        prompt = req.get("messages", [{}])[-1].get("content", "")
        max_tokens = int(req.get("max_tokens", 16))

        # Build the token list: oracle answer first, then filler tokens.
        ans = _oracle_answer(prompt) if CFG["oracle"] else "42"
        tokens = [ans] + [" ok"] * max(0, min(CFG["n_tokens"], max_tokens) - 1)

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        time.sleep(CFG["ttft"])  # simulated TTFT
        for i, tok in enumerate(tokens):
            if i > 0:
                time.sleep(CFG["tpot"])
            delta = {"choices": [{"delta": {"content": tok}}]}
            self._chunk(f"data: {json.dumps(delta)}\n\n".encode())
        self._chunk(b"data: [DONE]\n\n")
        self._chunk(b"")  # terminating 0-length chunk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--ttft", type=float, default=0.05)
    ap.add_argument("--tpot", type=float, default=0.005)
    ap.add_argument("--n-tokens", type=int, default=16)
    ap.add_argument("--no-oracle", action="store_true")
    args = ap.parse_args()
    CFG.update(ttft=args.ttft, tpot=args.tpot, n_tokens=args.n_tokens,
               oracle=not args.no_oracle)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"mock server on http://127.0.0.1:{args.port}  cfg={CFG}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
