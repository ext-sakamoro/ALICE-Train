#!/usr/bin/env python3
"""llm_bench.rs の `const SYSTEM_PROMPT: &str = "...";` を逐語抽出する

手コピーを避けるための機械抽出 Rust の文字列リテラルのうち本 const が使う機能だけを解釈する:
- `\\n\\` の行継続 (backslash が改行 + 直後の行頭 whitespace を食う)
- `\\n` / `\\t` / `\\"` / `\\\\` のエスケープ
"""
import re
import sys

src_path, out_path = sys.argv[1], sys.argv[2]
src = open(src_path, encoding="utf-8").read()

m = re.search(r'const SYSTEM_PROMPT: &str = "(.*?)";\n', src, re.DOTALL)
if not m:
    sys.exit("SYSTEM_PROMPT の const が見つからない")
raw = m.group(1)

out = []
i = 0
while i < len(raw):
    c = raw[i]
    if c != "\\":
        out.append(c)
        i += 1
        continue
    nxt = raw[i + 1]
    if nxt == "\n":
        # 行継続: 改行と直後の行頭 whitespace を捨てる
        i += 2
        while i < len(raw) and raw[i] in " \t":
            i += 1
        continue
    mapping = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\", "0": "\0"}
    if nxt not in mapping:
        sys.exit(f"未対応のエスケープ: \\{nxt}")
    out.append(mapping[nxt])
    i += 2

text = "".join(out)
open(out_path, "w", encoding="utf-8").write(text)
print(f"抽出: {len(text)} chars, {text.count(chr(10)) + 1} lines -> {out_path}")
print("--- 先頭 120 chars ---")
print(repr(text[:120]))
print("--- 末尾 80 chars ---")
print(repr(text[-80:]))
