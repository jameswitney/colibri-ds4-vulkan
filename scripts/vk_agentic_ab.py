#!/usr/bin/env python3
"""Agentic-use emulation A/B for the dsv4 Vulkan tier (review pass 2026-08-30).

Reuses scripts/vk_longgen_ab.py's machinery (arms, timing, byte-parity vs the
cpu arm) but with prompts that emulate REAL agentic workloads: multi-step tool
loops, error recovery, strict structured output, and code review. Each prompt
forces the model to emit machine-parseable tool-call JSON — the same class of
output an agent harness consumes — a good accuracy probe for the thresholded
GPU tier (REVIEW G12 envelope).

Usage (all flags forwarded to vk_longgen_ab.main):
  python3 scripts/vk_agentic_ab.py --max-tokens 64 --out .vk_bench_2h/agentic64
                                   [--arms cpu,vk-dense,vk-exp] [--model DIR]
                                   [--mem 12] [--prompts all]
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import vk_longgen_ab as lg  # noqa: E402

EXTRA = [
    ("agent-loop",
     "You are an autonomous agent with four tools: search_web(query: str), "
     "run_code(code: str), read_file(path: str), write_file(path: str, "
     "content: str). Task: (1) search the web for the current population of "
     "Tokyo; (2) verify the number against a second search result; (3) compute "
     "what fraction of Japan's population (125 million) that is; (4) write a "
     "one-paragraph summary to summary.md. Emit each tool call as exactly one "
     "line of JSON: {\"tool\": \"name\", \"args\": {\"...\": \"...\"}}. After "
     "each tool call you will receive a result; continue until the task is "
     "done, then output the final summary paragraph."),
    ("agent-recover",
     "You are an agent with three tools: fetch(url: str), "
     "parse_html(html: str), search(query: str). You called "
     "fetch(\"https://example.com/data\") and the result was HTTP 503 "
     "Service Unavailable. Emit the next sequence of tool calls you would "
     "make to recover: retry with an appropriate backoff, then fall back to "
     "a search-based alternative, then report to the user what you did and "
     "why. One JSON tool call per line, then a short final report."),
    ("json-schema",
     "Extract structured data. Input text: \"Order #1042: 3x Widget-A "
     "($9.99 each), 2x Gadget-B ($24.50 each), shipped 2026-08-15 to Berlin; "
     "grand total $79.97.\" Return ONLY a JSON object with exactly these "
     "fields: order_id (string), items (array of {name, qty, unit_price}), "
     "shipping_date (string, ISO format), destination (string), total "
     "(number). No prose, no markdown fences, valid JSON only."),
    ("code-review",
     "Review this Python code for bugs and list them precisely:\n"
     "```python\n"
     "def dedupe(items, seen=[]):\n"
     "    out = []\n"
     "    for i in range(1, len(items)+1):\n"
     "        if items[i] not in seen:\n"
     "            seen.append(items[i])\n"
     "            out.append(items[i])\n"
     "    return out\n"
     "```\n"
     "For each bug: name it, quote the offending line, explain the failure "
     "mode with a concrete example, and give the corrected line. Then "
     "summarize in one sentence."),
]

lg.PROMPTS = EXTRA
lg.main()
