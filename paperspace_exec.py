#!/usr/bin/env python3
"""Paperspace Jupyter Terminal 経由でコマンドを実行するヘルパー。"""
import websocket
import json
import sys
import time
import re
import os

WS_URL = os.environ.get("PAPERSPACE_WS", "wss://n6doryrfnz.clg07azjl.paperspacegradient.com/terminals/websocket/1?token=0b5068a7eeda1e0a2b7bf723278720e2")
MARKER = "__ALICE_CMD_DONE__"

def run_cmd(cmd: str, timeout: int = 120) -> str:
    ws = websocket.create_connection(WS_URL, timeout=15)

    # Drain any buffered output first (history replay)
    while True:
        try:
            ws.settimeout(2)
            ws.recv()
        except:
            break

    # Send command with unique marker
    ts = int(time.time() * 1000)
    start_marker = f"__START_{ts}__"
    # Wrap in subshell to handle & and other special chars
    escaped_cmd = cmd.replace("'", "'\\''")
    full_cmd = f"echo {start_marker} && bash -c '{escaped_cmd}'\necho {MARKER}$?\r"
    ws.send(json.dumps(["stdin", full_cmd]))

    output = []
    deadline = time.time() + timeout
    capturing = False
    while time.time() < deadline:
        try:
            ws.settimeout(5)
            msg = ws.recv()
            data = json.loads(msg)
            if data[0] == "stdout":
                text = data[1]
                if start_marker in text:
                    capturing = True
                    # Only keep text after the start marker
                    idx = text.index(start_marker) + len(start_marker)
                    text = text[idx:]
                if capturing:
                    output.append(text)
                    joined = "".join(output)
                    if MARKER in joined:
                        # Drain a bit more
                        time.sleep(0.3)
                        while True:
                            try:
                                ws.settimeout(0.3)
                                msg = ws.recv()
                                data = json.loads(msg)
                                if data[0] == "stdout":
                                    output.append(data[1])
                            except:
                                break
                        break
        except websocket.WebSocketTimeoutException:
            continue
        except Exception as e:
            output.append(f"\n[ERROR: {e}]\n")
            break

    ws.close()

    # Clean ANSI escape codes
    raw = "".join(output)
    clean = re.sub(r'\x1b\[[^a-zA-Z]*[a-zA-Z]', '', raw)
    clean = re.sub(r'\x1b\][^\x07]*\x07', '', clean)
    clean = re.sub(r'\[[\?0-9;]*[a-zA-Z]', '', clean)

    # Remove the marker lines from output
    lines = clean.split('\n')
    result_lines = []
    for line in lines:
        if MARKER in line or start_marker in line:
            continue
        result_lines.append(line)
    return '\n'.join(result_lines).strip()

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "echo hello"
    timeout = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    print(run_cmd(cmd, timeout))
