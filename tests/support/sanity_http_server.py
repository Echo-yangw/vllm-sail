# SPDX-License-Identifier: Apache-2.0
"""Strict external serving double for the sanity runner's HTTP/lifecycle tests."""

import argparse
import json
import signal
from http.server import BaseHTTPRequestHandler, HTTPServer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", default="ok")
    args = parser.parse_args()

    class Handler(BaseHTTPRequestHandler):
        def reply(self, value):
            data = json.dumps(value).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/v1/models":
                name = "unrelated-service" if args.mode == "stale" else args.model
                self.reply({"data": [{"id": name}]})
            elif self.path == "/metrics":
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"vllm:generation_tokens_total 2\n")
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                self.send_error(404)
                return
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if body["model"] != args.model or body["temperature"] != 0:
                self.send_error(400)
                return
            if args.mode == "http-error":
                self.send_error(500)
                return
            finish = "length" if args.mode == "truncated" else "stop"
            content = "43" if args.mode == "wrong" else "42"
            if args.mode in ("inline-think", "inline-missing-stream-boundary"):
                content = "Compute 17 + 25 = 42.</think>42"
            if body.get("stream"):
                if args.mode == "minimax-think":
                    content = "<mm:think>Compute 17 + 25 = 42.</mm:think>42"
                elif args.mode == "minimax-missing-final":
                    content = "<mm:think>The result is 42.</mm:think>"
                if args.mode == "inline-missing-stream-boundary":
                    content = "42"
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                pieces = [content]
                if args.mode == "inline-think":
                    split = content.index("</think>") + 4
                    pieces = [content[:split], content[split:-2], content[-2:]]
                elif args.mode == "minimax-think":
                    split = content.index("</mm:think>") + 5
                    pieces = [content[:3], content[3:split], content[split:-2], "42"]
                for index, piece in enumerate(pieces):
                    event = {
                        "model": args.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": piece},
                                "finish_reason": finish
                                if index == len(pieces) - 1
                                else None,
                            }
                        ],
                    }
                    self.wfile.write(b"data: " + json.dumps(event).encode() + b"\n\n")
                if args.mode != "missing-done":
                    self.wfile.write(b"data: [DONE]\n\n")
            else:
                self.reply(
                    {
                        "model": args.model,
                        "choices": [
                            {"message": {"content": content}, "finish_reason": finish}
                        ],
                    }
                )
            if args.mode == "log-error":
                print("ERROR injected serving failure", flush=True)

    def stop(signum, frame):
        if args.mode == "shutdown-error":
            print("ERROR injected shutdown failure", flush=True)
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, stop)
    with HTTPServer(("127.0.0.1", args.port), Handler) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
