#!/usr/bin/env python3
"""Long/complex-prompt A/B for the dsv4 Vulkan tier (M5 follow-up, 2026-08-30).

Accuracy (byte parity vs the CPU chain) and performance (wall / tok/s) on
longer generations and more complicated prompts — tool-use-style structured
output, code generation, multi-step math reasoning, plus the G12 control
(photosynthesis) and a long (~400-token) multi-part prompt that exercises the
M4 batched prefill ops on GPU.

Arms (all greedy, same CLI, same checkpoint):
  cpu       — tier off (COLI_DSV4_VK_DENSE unset)  = the accuracy baseline
  vk-dense  — COLI_DSV4_VK_DENSE=1                 = dense fp8 tier (D7 drop)
  vk-exp    — COLI_DSV4_VK_DENSE=1 + EXPERTS=1     = dense + fp4 expert tier

For each (prompt, arm): run the CLI, capture stdout/stderr/wall, extract the
generated text (stdout minus TUNE lines), compare byte-for-byte vs the cpu arm,
and report wall + tok/s. A divergence beyond the documented parity envelope
(REVIEW G12: verified ≤16 tokens, 18+ after the fp64 fix) is expected on some
prompts and reported as a finding, not a regression — unless the dense and exp
arms diverge from EACH OTHER (which would be a tier bug, not envelope drift).

Usage:
  python3 scripts/vk_longgen_ab.py [--max-tokens 64] [--model DIR] [--mem 12]
                                   [--arms cpu,vk-dense,vk-exp] [--out DIR]
Env overrides: COLI_MODEL, COLI_MAX_TOKENS, COLI_MEMORY_GB, COLI_BIN.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

DEFAULT_MODEL = os.environ.get("COLI_MODEL", "models/DeepSeek-V4-Flash")

PROMPTS = [
    ("tool-use", "You are a helpful assistant with access to these tools: "
     "search(query: str), calculator(expression: str), get_weather(city: str). "
     "The user asks: \"What's the weather in Paris today, and what is 17 times 23?\" "
     "First emit the sequence of tool calls you would make, one JSON object per "
     "line, using exactly this schema: {\"tool\": \"name\", \"args\": {\"key\": "
     "\"value\"}}. Then write the final answer to the user."),
    ("code-lru", "Write a complete Python implementation of an LRU cache with "
     "O(1) get and put operations using only collections.OrderedDict. Include "
     "type hints, docstrings, and handle the capacity-0 edge case. Then "
     "explain the time complexity of each operation and why OrderedDict "
     "supports O(1) move_to_end."),
    ("math-trains", "A freight train leaves City A at 8:30 AM traveling at 60 "
     "miles per hour toward City B, which is 240 miles away. A passenger train "
     "leaves City B at 9:00 AM traveling at 75 miles per hour toward City A. "
     "At what time do the two trains meet, and how far from City A is the "
     "meeting point? Show each step of your calculation."),
    ("photosynthesis", "Write a short paragraph explaining how photosynthesis "
     "works. Include the roles of sunlight, water, and carbon dioxide."),
    ("long-multi", "You are an assistant helping a product manager prepare a "
     "quarterly review. Write a structured report covering all of the "
     "following, in order, with clear section headers: (1) a summary of the "
     "quarter's three biggest wins, each with a one-sentence rationale; (2) "
     "two risks that could derail next quarter, each with a proposed "
     "mitigation; (3) a prioritized list of five initiatives for next "
     "quarter, ranked by expected impact divided by effort, with a short "
     "justification for the top two; (4) a closing paragraph that ties the "
     "wins, risks, and initiatives into a single narrative. The report must "
     "be self-contained and addressed to a general audience of executives."),
]

ARMS = {
    "cpu": {},
    "vk-dense": {"COLI_DSV4_VK_DENSE": "1"},
    "vk-exp": {"COLI_DSV4_VK_DENSE": "1", "COLI_DSV4_VK_EXPERTS": "1"},
}


@dataclass
class Run:
    arm: str
    prompt: str
    wall_s: float = 0.0
    rc: int = -1
    text: str = ""
    stderr_tail: str = ""
    timing: dict = field(default_factory=dict)


def extract_text(stdout: str) -> str:
    """Generated text = stdout minus the TUNE wall line(s)."""
    lines = [ln for ln in stdout.splitlines() if not ln.startswith("TUNE")]
    return "\n".join(lines).strip()


def run_cli(binary, model, prompt, max_tokens, mem_gb, arm_env, timeout):
    env = dict(os.environ)
    env.update({
        "COLI_TEMP": "0",
        "KVSAVE": "0",
        "OMP_NUM_THREADS": "8",
    })
    env.update(arm_env)
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            [binary, model, prompt, "--max-tokens", str(max_tokens),
             "--memory-gb", str(mem_gb)],
            env=env, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=timeout)
        wall = time.monotonic() - t0
        return proc.returncode, wall, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as e:
        wall = time.monotonic() - t0
        return -9, wall, (e.stdout or ""), (e.stderr or "")


def parse_timing(stderr: str) -> dict:
    out = {}
    for ln in stderr.splitlines():
        if ln.startswith("timing "):
            for tok in ln[len("timing "):].split():
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    if v.endswith("s"):
                        v = v[:-1]  # "15.556s"
                    try:
                        out[k] = float(v)
                    except ValueError:
                        out[k] = v
    return out


def first_diff_char(a: str, b: str):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n if len(a) != len(b) else -1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=os.environ.get("COLI_MODEL", DEFAULT_MODEL))
    ap.add_argument("--max-tokens", type=int,
                    default=int(os.environ.get("COLI_MAX_TOKENS", "64")))
    ap.add_argument("--mem", type=int, default=int(os.environ.get("COLI_MEMORY_GB", "12")))
    ap.add_argument("--arms", default="cpu,vk-dense,vk-exp")
    ap.add_argument("--binary", default=os.environ.get("COLI_BIN",
                    os.path.join(ROOT, "c", "deepseek_v4")))
    ap.add_argument("--out", default=os.path.join(ROOT, ".vk_longgen"))
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--prompts", default="all",
                    help="comma list of prompt names or 'all'")
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    want = None if args.prompts == "all" else set(args.prompts.split(","))
    prompts = [(n, p) for (n, p) in PROMPTS if want is None or n in want]

    if not os.path.isdir(args.model):
        print(f"model dir not found: {args.model}", file=sys.stderr)
        sys.exit(2)
    if not os.access(args.binary, os.X_OK):
        print(f"binary not executable: {args.binary}", file=sys.stderr)
        sys.exit(2)

    os.makedirs(args.out, exist_ok=True)
    print(f"binary={args.binary}\nmodel={args.model}\nmax_tokens={args.max_tokens}"
          f"\nmem={args.mem} GiB\narms={arms}\nprompts={[n for n, _ in prompts]}")

    results = {}
    for name, prompt in prompts:
        runs = {}
        for arm in arms:
            rc, wall, stdout, stderr = run_cli(
                args.binary, args.model, prompt, args.max_tokens, args.mem,
                ARMS[arm], args.timeout)
            r = Run(arm=arm, prompt=name, wall_s=wall, rc=rc,
                    text=extract_text(stdout), stderr_tail=stderr[-800:],
                    timing=parse_timing(stderr))
            runs[arm] = r
            n = r.timing.get("after_first", 0.0)
            tps = (args.max_tokens / n) if n > 0 else 0.0
            print(f"  [{name:14s}][{arm:9s}] rc={rc} wall={wall:7.1f}s "
                  f"ttft={r.timing.get('time_to_first_token', -1):7.1f}s "
                  f"tok/s={tps:5.2f} text_len={len(r.text)}")
            with open(os.path.join(args.out, f"{name}.{arm}.txt"), "w") as f:
                f.write(r.text)
            with open(os.path.join(args.out, f"{name}.{arm}.stderr"), "w") as f:
                f.write(stderr)
        results[name] = runs

    print("\n=== ACCURACY (byte parity vs cpu arm) ===")
    for name, runs in results.items():
        cpu = runs.get("cpu")
        if cpu is None:
            continue
        for arm in ("vk-dense", "vk-exp"):
            r = runs.get(arm)
            if r is None:
                continue
            if r.text == cpu.text:
                print(f"  [{name:14s}][{arm:9s}] IDENTICAL ({len(cpu.text)} chars)")
            else:
                d = first_diff_char(cpu.text, r.text)
                print(f"  [{name:14s}][{arm:9s}] DIFFERS  (first diff at char "
                      f"{d}; cpu_len={len(cpu.text)} vk_len={len(r.text)})")
                if d >= 0:
                    print(f"      cpu: ...{cpu.text[max(0, d-60):d+120]!r}")
                    print(f"      vk : ...{r.text[max(0, d-60):d+120]!r}")

    print("\n=== DENSE vs EXP consistency (tier-internal) ===")
    for name, runs in results.items():
        if "vk-dense" in runs and "vk-exp" in runs:
            a, b = runs["vk-dense"].text, runs["vk-exp"].text
            tag = "IDENTICAL" if a == b else "DIFFERS"
            print(f"  [{name:14s}] dense vs exp: {tag}")

    print("\n=== PERFORMANCE ===")
    hdr = f"  {'prompt':14s} {'arm':9s} {'wall_s':>8s} {'tok/s':>7s} {'ttft_s':>8s}"
    print(hdr)
    for name, runs in results.items():
        for arm, r in runs.items():
            n = r.timing.get("after_first", 0.0)
            tps = (args.max_tokens / n) if n > 0 else 0.0
            print(f"  {name:14s} {arm:9s} {r.wall_s:8.1f} {tps:7.2f} "
                  f"{r.timing.get('time_to_first_token', -1):8.1f}")

    summary = os.path.join(args.out, "summary.json")
    with open(summary, "w") as f:
        json.dump({n: {a: {"wall_s": r.wall_s, "rc": r.rc,
                           "timing": r.timing, "text": r.text}
                       for a, r in rs.items()}
                   for n, rs in results.items()}, f, indent=2)
    print(f"\nresults in {args.out}/ (summary.json + per-run text/stderr)")


if __name__ == "__main__":
    main()
