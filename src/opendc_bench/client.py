"""Async OpenAI-compatible streaming client, dependency-free (stdlib asyncio).

Implements the spec 2.9 measurement contract precisely:
  * request start time: monotonic, captured immediately before send;
  * TTFT: time to the first NON-EMPTY delta.content (metadata-only / empty
    chunks do not count);
  * TPOT: (t_end - t_first) / max(y_i - 1, 1), with y_i re-tokenized;
  * E2E: time until the final event is parsed and the stream closes cleanly;
  * timeouts / malformed / incomplete -> success=False.

Token accounting re-tokenizes the concatenated output with the declared
tokenizer (no server-side trust by default).
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import List, Optional
from urllib.parse import urlparse

from .metrics import RequestResult


class StreamClient:
    def __init__(self, base_url: str, model: str, tokenizer, request_timeout: float = 300.0):
        u = urlparse(base_url)
        if u.scheme not in ("http", ""):
            raise ValueError("only http endpoints supported (vLLM/SGLang/TGI serve http internally)")
        self.host = u.hostname
        self.port = u.port or 80
        self.path = "/v1/chat/completions"
        self.model = model
        self.tok = tokenizer
        self.timeout = request_timeout

    async def _read_headers(self, reader: asyncio.StreamReader):
        status_line = await reader.readline()
        if not status_line:
            raise ConnectionError("empty response")
        # b"HTTP/1.1 200 OK" -> 200
        parts = status_line.split()
        status = int(parts[1]) if len(parts) >= 2 and parts[1].isdigit() else 0
        headers = {}
        while True:
            line = await reader.readline()
            if line in (b"\r\n", b"\n", b""):
                break
            k, _, v = line.decode("latin1").partition(":")
            headers[k.strip().lower()] = v.strip()
        return status, headers

    async def _iter_chunked(self, reader: asyncio.StreamReader):
        """Yield raw body bytes from a chunked transfer-encoded stream."""
        while True:
            size_line = await reader.readline()
            if not size_line:
                return
            try:
                size = int(size_line.strip().split(b";")[0], 16)
            except ValueError:
                return
            if size == 0:
                await reader.readline()  # trailing CRLF
                return
            data = await reader.readexactly(size)
            await reader.readexactly(2)  # CRLF
            yield data

    async def complete(self, rec: dict) -> RequestResult:
        """Send one streaming request built from a dataset record `rec`
        (must carry: id, workload, prompt, max_output_tokens)."""
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": rec["prompt"]}],
            "max_tokens": rec["max_output_tokens"],
            "temperature": 0.0,
            "top_p": 1.0,
            "stream": True,
        }).encode("utf-8")
        req = (
            f"POST {self.path} HTTP/1.1\r\n"
            f"Host: {self.host}:{self.port}\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\n"
            f"Accept: text/event-stream\r\n"
            f"Connection: close\r\n\r\n"
        ).encode("latin1") + body

        n_input = len(self.tok.encode(rec["prompt"]))
        start = time.monotonic()
        t_first: Optional[float] = None
        out_parts: List[str] = []
        success = False
        error: Optional[str] = None

        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(self.host, self.port), timeout=self.timeout)
            writer.write(req)
            await writer.drain()

            status, headers = await asyncio.wait_for(
                self._read_headers(reader), timeout=self.timeout)

            if status >= 400:
                # read the error body to classify (OOM vs context vs server)
                body = b""
                try:
                    body = await asyncio.wait_for(reader.read(4096), timeout=5)
                except Exception:
                    pass
                low = body.lower()
                if b"out of memory" in low or b"oom" in low or b"hip out" in low or b"cuda out" in low:
                    error = "oom"
                elif status >= 500:
                    error = "server_error"
                else:
                    error = "client_error"  # 4xx: context-length, bad request, etc.
                writer.close()
                t_end = time.monotonic()
                return RequestResult(
                    id=rec["id"], workload=rec["workload"], start_time=start,
                    ttft=None, tpot=None, e2e=None, completion_time=t_end,
                    n_output_tokens=0, n_input_tokens=n_input, output_text="",
                    success=False, error=error)

            async def consume():
                nonlocal t_first, success
                buf = b""
                source = (self._iter_chunked(reader)
                          if "chunked" in headers.get("transfer-encoding", "")
                          else self._iter_until_close(reader))
                async for chunk in source:
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        line = line.strip()
                        if not line.startswith(b"data:"):
                            continue
                        payload = line[len(b"data:"):].strip()
                        if payload == b"[DONE]":
                            success = True
                            return
                        try:
                            obj = json.loads(payload)
                        except json.JSONDecodeError:
                            continue
                        delta = (obj.get("choices") or [{}])[0].get("delta", {})
                        piece = delta.get("content")
                        if piece:
                            if t_first is None:
                                t_first = time.monotonic()
                            out_parts.append(piece)

            await asyncio.wait_for(consume(), timeout=self.timeout)
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
        except asyncio.TimeoutError:
            error = "timeout"
        except Exception as e:  # connection reset, malformed, etc.
            error = "conn_error"

        t_end = time.monotonic()
        text = "".join(out_parts)
        n_out = len(self.tok.encode(text)) if text else 0

        ttft = (t_first - start) if t_first is not None else None
        e2e = (t_end - start) if success else None
        tpot = None
        if success and t_first is not None and n_out > 0:
            tpot = (t_end - t_first) / max(n_out - 1, 1)

        # A "successful" run must have produced at least one token.
        ok = success and n_out > 0 and error is None
        if not ok and error is None:
            error = "conn_error"  # stream closed without [DONE] / no tokens
        return RequestResult(
            id=rec["id"], workload=rec["workload"], start_time=start,
            ttft=ttft, tpot=tpot, e2e=e2e, completion_time=t_end,
            n_output_tokens=n_out, n_input_tokens=n_input, output_text=text,
            success=ok, error=error,
        )

    async def _iter_until_close(self, reader: asyncio.StreamReader):
        while True:
            data = await reader.read(65536)
            if not data:
                return
            yield data
