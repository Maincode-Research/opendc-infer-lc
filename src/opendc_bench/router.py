"""Minimal dependency-free round-robin reverse proxy.

For the multi-node scaling study (spec 2.7 Efficiency(N)) we run one serving
replica per node and present a SINGLE OpenAI-compatible endpoint to the load
generator. This router round-robins each incoming request across the backends
and streams the response back byte-for-byte (no SSE parsing — pure passthrough,
so it adds negligible latency and never becomes the bottleneck).

Assumes the load generator uses one request per connection (Connection: close),
which opendc_bench.client does.

Run:  python -m opendc_bench.router --port 8000 --backends node01:8001,node02:8001
"""
from __future__ import annotations

import argparse
import asyncio
import itertools
from typing import List, Tuple


async def _read_http_request(reader: asyncio.StreamReader) -> bytes:
    """Read a full HTTP/1.1 request (head + Content-Length body) as raw bytes."""
    head = b""
    while b"\r\n\r\n" not in head:
        chunk = await reader.read(65536)
        if not chunk:
            break
        head += chunk
    if not head:
        return b""
    header_blob, _, rest = head.partition(b"\r\n\r\n")
    clen = 0
    for line in header_blob.split(b"\r\n"):
        if line.lower().startswith(b"content-length:"):
            clen = int(line.split(b":", 1)[1].strip())
            break
    body = rest
    while len(body) < clen:
        chunk = await reader.read(65536)
        if not chunk:
            break
        body += chunk
    return header_blob + b"\r\n\r\n" + body


class Router:
    def __init__(self, backends: List[Tuple[str, int]]):
        self.backends = backends
        self._rr = itertools.cycle(range(len(backends)))

    async def handle(self, c_reader: asyncio.StreamReader, c_writer: asyncio.StreamWriter):
        try:
            req = await _read_http_request(c_reader)
            if not req:
                c_writer.close()
                return
            host, port = self.backends[next(self._rr)]
            b_reader, b_writer = await asyncio.open_connection(host, port)
            b_writer.write(req)
            await b_writer.drain()
            # stream backend -> client until backend closes
            while True:
                data = await b_reader.read(65536)
                if not data:
                    break
                c_writer.write(data)
                await c_writer.drain()
            b_writer.close()
        except Exception:
            pass
        finally:
            try:
                c_writer.close()
            except Exception:
                pass

    async def serve(self, host: str, port: int):
        server = await asyncio.start_server(self.handle, host, port)
        addrs = ", ".join(f"{h}:{p}" for h, p in self.backends)
        print(f"router on {host}:{port} -> [{addrs}]", flush=True)
        async with server:
            await server.serve_forever()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--backends", required=True,
                    help="comma-separated host:port list")
    args = ap.parse_args()
    backends = []
    for b in args.backends.split(","):
        h, _, p = b.strip().rpartition(":")
        backends.append((h, int(p)))
    asyncio.run(Router(backends).serve(args.host, args.port))


if __name__ == "__main__":
    main()
